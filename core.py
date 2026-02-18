
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Headless-safe plotting for servers/CI
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    import cvxpy as cp  # type: ignore
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models, callbacks  # type: ignore
    HAS_TF = True
except Exception:
    HAS_TF = False


@dataclass
class PipelineConfig:
    date_col: str = "Date"
    min_coverage: float = 0.70
    price_col_candidates: Tuple[str, ...] = ("PX_LAST", "Close", "Price", "Adj Close", "Adj_Close")

    rebalance_freq: str = "W-FRI"
    estimation_window_weeks: int = 60

    cov_shrink_lam: float = 0.10

    long_only: bool = True
    weight_cap: Optional[float] = 0.30

    tc_bps: float = 10.0  # bps * L1 turnover
    rf_annual: float = 0.0

    regime_q_low: float = 1/3
    regime_q_high: float = 2/3

    lstm_lookback: int = 12
    lstm_epochs: int = 25
    lstm_batch: int = 32
    seed: int = 42


@dataclass
class PipelineOutputs:
    outdir: Path
    tables: Dict[str, Path]
    figures: Dict[str, Path]
    artifacts: Dict[str, Path]
    meta: Dict[str, str]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_col(cols: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    m = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None


def load_multisheet_prices(xlsx_path: Path, cfg: PipelineConfig) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    series: Dict[str, pd.Series] = {}

    for sheet, df in sheets.items():
        if df is None or df.empty:
            continue
        if cfg.date_col not in df.columns:
            continue
        pcol = _find_col(list(df.columns), cfg.price_col_candidates)
        if pcol is None:
            continue

        d = df[[cfg.date_col, pcol]].copy()
        d[cfg.date_col] = pd.to_datetime(d[cfg.date_col], errors="coerce")
        d = d.dropna(subset=[cfg.date_col]).sort_values(cfg.date_col)
        d = d.drop_duplicates(subset=[cfg.date_col], keep="last").set_index(cfg.date_col)
        s = pd.to_numeric(d[pcol], errors="coerce").astype(float)
        s.name = str(sheet)
        series[str(sheet)] = s

    if len(series) < 2:
        raise ValueError("Need at least 2 valid price series (Date + price column) across sheets.")

    prices = pd.concat(series.values(), axis=1).sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    keep = [c for c in prices.columns if float(prices[c].notna().mean()) >= cfg.min_coverage]
    prices = prices[keep]
    if prices.shape[1] < 2:
        raise ValueError("After coverage filtering, fewer than 2 series remain.")
    return prices


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = np.log(prices).diff()
    return r.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def annualize_daily(log_r: np.ndarray, freq: int = 252) -> Tuple[float, float, float]:
    x = np.asarray(log_r, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    ann_ret = float(np.exp(np.mean(x) * freq) - 1.0)
    ann_vol = float(np.std(x) * np.sqrt(freq))
    sharpe = np.nan if ann_vol < 1e-12 else float((np.mean(x) / (np.std(x) + 1e-12)) * np.sqrt(freq))
    return ann_ret, ann_vol, sharpe


def equity_curve_from_log_returns(log_r: pd.Series) -> pd.Series:
    lr = pd.Series(log_r).fillna(0.0)
    return np.exp(lr.cumsum())


def drawdown_from_equity_curve(v: pd.Series) -> pd.Series:
    peak = v.cummax()
    return v / peak - 1.0


def rolling_sharpe_weekly(log_r_weekly: pd.Series, window_weeks: int, rf_annual: float = 0.0) -> pd.Series:
    x = log_r_weekly.values.astype(float)
    out = np.full_like(x, np.nan, dtype=float)
    rf_w = rf_annual / 52.0
    for i in range(window_weeks - 1, len(x)):
        seg = x[i - window_weeks + 1 : i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size < max(8, window_weeks // 3):
            continue
        ex = seg - rf_w
        sd = np.std(ex)
        out[i] = np.nan if sd < 1e-12 else float(np.mean(ex) / sd * np.sqrt(52))
    return pd.Series(out, index=log_r_weekly.index, name="rolling_sharpe")


def shrink_covariance(cov: np.ndarray, lam: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    diag = np.diag(np.diag(cov))
    return (1.0 - lam) * cov + lam * diag


def turnover_l1(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(np.sum(np.abs(w_new - w_old)))


def solve_min_var(cov: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    n = cov.shape[0]
    cov = cov + 1e-10 * np.eye(n)

    if HAS_CVXPY:
        w = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, cov))
        cons = [cp.sum(w) == 1]
        if cfg.long_only:
            cons.append(w >= 0)
        if cfg.weight_cap is not None:
            cons.append(w <= float(cfg.weight_cap))
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if w.value is not None:
                wv = np.asarray(w.value).reshape(-1)
                if cfg.long_only:
                    wv = np.clip(wv, 0, None)
                return wv / (wv.sum() + 1e-12)
        except Exception:
            pass

    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    wv = inv @ ones
    if cfg.long_only:
        wv = np.clip(wv, 0, None)
    if cfg.weight_cap is not None:
        wv = np.minimum(wv, float(cfg.weight_cap))
    return wv / (wv.sum() + 1e-12)


def solve_risk_parity(cov: np.ndarray, iters: int = 1500, lr: float = 0.05, tol: float = 1e-7) -> np.ndarray:
    n = cov.shape[0]
    cov = cov + 1e-10 * np.eye(n)
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w
        rc = w * mrc
        tgt = rc.mean()
        grad = rc - tgt
        w2 = w - lr * grad
        w2 = np.clip(w2, 0, None)
        w2 = w2 / (w2.sum() + 1e-12)
        if np.linalg.norm(w2 - w) < tol:
            w = w2
            break
        w = w2
    return w


def solve_mv_target(mu: np.ndarray, cov: np.ndarray, mu_star: float, cfg: PipelineConfig) -> np.ndarray:
    n = cov.shape[0]
    cov = cov + 1e-10 * np.eye(n)

    if HAS_CVXPY:
        w = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, cov))
        cons = [cp.sum(w) == 1, mu @ w == float(mu_star)]
        if cfg.long_only:
            cons.append(w >= 0)
        if cfg.weight_cap is not None:
            cons.append(w <= float(cfg.weight_cap))
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if w.value is not None:
                wv = np.asarray(w.value).reshape(-1)
                if cfg.long_only:
                    wv = np.clip(wv, 0, None)
                return wv / (wv.sum() + 1e-12)
        except Exception:
            pass

    return solve_min_var(cov, cfg)


def compute_weekly_features(returns_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    weekly_sector = returns_daily.resample(freq).sum(min_count=3).dropna(how="all")
    cs_ret = weekly_sector.mean(axis=1)

    vol_daily = returns_daily.rolling(21, min_periods=10).std().mean(axis=1).dropna()
    vol_w_last = vol_daily.resample(freq).last()
    vol_w_mean = vol_daily.resample(freq).mean()

    mom_8w = cs_ret.rolling(8, min_periods=4).mean()
    rev_2w = -cs_ret.rolling(2, min_periods=1).mean()

    feats = pd.DataFrame({
        "weekly_return_cs": cs_ret,
        "weekly_vol_mean": vol_w_mean,
        "mom_8w": mom_8w,
        "rev_2w": rev_2w,
        "mkt_vol_weekly": vol_w_last,
    }).dropna()
    return feats


def rule_based_regimes(mkt_vol_weekly: pd.Series, q_low: float, q_high: float) -> Tuple[pd.Series, Tuple[float, float]]:
    t1 = float(mkt_vol_weekly.quantile(q_low))
    t2 = float(mkt_vol_weekly.quantile(q_high))

    def lab(v: float) -> int:
        if v <= t1:
            return 0
        if v <= t2:
            return 1
        return 2

    y = mkt_vol_weekly.apply(lab).astype(int)
    y.name = "regime"
    return y, (t1, t2)


def _make_sequences(X: pd.DataFrame, y: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    y2 = y.reindex(X.index).dropna()
    X2 = X.loc[y2.index]
    Xv, yv, idx = [], [], []
    for i in range(lookback, len(X2)):
        Xv.append(X2.iloc[i - lookback : i].values)
        yv.append(int(y2.iloc[i]))
        idx.append(X2.index[i])
    return np.asarray(Xv), np.asarray(yv), pd.DatetimeIndex(idx)


def lstm_regime_classifier(feats: pd.DataFrame, y: pd.Series, cfg: PipelineConfig) -> Tuple[Optional[pd.Series], Optional[np.ndarray], Dict[str, float]]:
    metrics = {"accuracy": np.nan, "macro_f1": np.nan}
    if not HAS_TF:
        return None, None, metrics

    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Xraw, yraw, idx = _make_sequences(feats, y, cfg.lstm_lookback)
    if len(Xraw) < 80:
        return None, None, metrics

    split = int(0.8 * len(Xraw))
    Xtr, Xte = Xraw[:split], Xraw[split:]
    ytr, yte = yraw[:split], yraw[split:]

    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-8

    def norm(A: np.ndarray) -> np.ndarray:
        return (A - mu) / sd

    Xtr, Xte, Xall = norm(Xtr), norm(Xte), norm(Xraw)

    model = models.Sequential([
        layers.Input(shape=(Xtr.shape[1], Xtr.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=cfg.lstm_epochs, batch_size=cfg.lstm_batch, verbose=0, callbacks=cb)

    yhat = model.predict(Xte, verbose=0).argmax(axis=1)
    acc = float((yhat == yte).mean())

    def f1(k: int) -> float:
        tp = np.sum((yhat == k) & (yte == k))
        fp = np.sum((yhat == k) & (yte != k))
        fn = np.sum((yhat != k) & (yte == k))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        return float(2 * prec * rec / (prec + rec + 1e-12))

    macro_f1 = float(np.mean([f1(0), f1(1), f1(2)]))
    metrics = {"accuracy": acc, "macro_f1": macro_f1}

    cm = np.zeros((3, 3), dtype=int)
    for yt, yh in zip(yte, yhat):
        cm[int(yt), int(yh)] += 1

    pred_all = model.predict(Xall, verbose=0).argmax(axis=1)
    pred = pd.Series(pred_all, index=idx, name="pred_regime").astype(int)

    return pred, cm, metrics


def backtest(returns_daily: pd.DataFrame, policy: pd.Series, cfg: PipelineConfig) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    weekly = returns_daily.resample(cfg.rebalance_freq).sum(min_count=3).dropna(how="all")
    weekly = weekly.dropna(axis=1, how="any")
    sectors = list(weekly.columns)
    n = len(sectors)

    policy = policy.reindex(weekly.index).ffill().dropna()

    w_prev = np.ones(n) / n
    port, weights_rows, applied = [], [], []

    for i in range(cfg.estimation_window_weeks, len(weekly) - 1):
        d = weekly.index[i]
        d_next = weekly.index[i + 1]
        pol = int(policy.loc[d])

        window = weekly.iloc[i - cfg.estimation_window_weeks : i]
        mu = np.nanmean(window.values, axis=0)
        cov = np.cov(window.values.T)
        cov = shrink_covariance(cov, cfg.cov_shrink_lam)
        mu_star = float(mu.mean())

        if pol == 0:
            w = solve_mv_target(mu, cov, mu_star, cfg)
            pol_name = "Mean-Variance"
        elif pol == 1:
            w = solve_risk_parity(cov)
            pol_name = "Risk-Parity"
        else:
            w = solve_min_var(cov, cfg)
            pol_name = "Minimum-Variance"

        tc = (cfg.tc_bps / 10000.0) * turnover_l1(w, w_prev)
        r_next = float(weekly.loc[d_next].values @ w) - tc

        port.append((d_next, r_next))
        weights_rows.append((d, pol_name, *w))
        applied.append((d, pol))
        w_prev = w

    port_s = pd.Series([x[1] for x in port], index=pd.DatetimeIndex([x[0] for x in port]), name="portfolio_log_return").sort_index()
    wdf = pd.DataFrame(weights_rows, columns=["Date", "Policy"] + sectors).set_index("Date")
    applied_s = pd.Series({d: p for d, p in applied}, name="applied_regime").sort_index()
    return port_s, wdf, applied_s


def perf_metrics_weekly(port: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    x = port.dropna().values.astype(float)
    if x.size == 0:
        return {"AnnReturn": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    ann_ret = float(np.exp(np.mean(x) * 52) - 1.0)
    ann_vol = float(np.std(x) * np.sqrt(52))
    rf_w = rf_annual / 52.0
    ex = x - rf_w
    sd = np.std(ex)
    sharpe = np.nan if sd < 1e-12 else float(np.mean(ex) / sd * np.sqrt(52))

    eq = equity_curve_from_log_returns(pd.Series(x, index=port.dropna().index))
    dd = drawdown_from_equity_curve(eq)
    maxdd = float(dd.min()) if len(dd) else np.nan
    return {"AnnReturn": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "MaxDD": maxdd}


def _plot_risk_return_scatter(returns_daily: pd.DataFrame, outdir: Path) -> Path:
    rows = []
    for c in returns_daily.columns:
        ar, av, sh = annualize_daily(returns_daily[c].dropna().values)
        rows.append({"Sector": c, "AnnReturn": ar, "AnnVol": av, "Sharpe": sh})
    df = pd.DataFrame(rows).dropna()

    fp = outdir / "Figure_4.1_Risk_Return_Scatter.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(df["AnnVol"], df["AnnReturn"], s=80)
    for _, r in df.iterrows():
        plt.text(r["AnnVol"], r["AnnReturn"], str(r["Sector"]), fontsize=8, ha="right", va="bottom")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Risk–Return Characteristics (NGX Sectors)")
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()

    (outdir / "Table_Sector_Risk_Return_Summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    return fp


def _plot_corr_heatmap(returns_daily: pd.DataFrame, outdir: Path) -> Path:
    corr = returns_daily.dropna(how="any").corr()
    fp = outdir / "Figure_4.2_Correlation_Heatmap.png"
    plt.figure(figsize=(9, 7))
    plt.imshow(corr.values, interpolation="nearest", aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Correlation Heatmap (NGX Sector Returns)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()

    corr.to_csv(outdir / "Table_Correlation_Matrix.csv")
    return fp


def _plot_confusion_matrix(cm: np.ndarray, outdir: Path) -> Path:
    fp = outdir / "Figure_4.3_Confusion_Matrix.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Regime Classifier)")
    plt.xticks([0, 1, 2], ["Low", "Med", "High"])
    plt.yticks([0, 1, 2], ["Low", "Med", "High"])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()
    return fp


def _plot_regimes(applied: pd.Series, outdir: Path) -> Path:
    fp = outdir / "Figure_4.4_Applied_Regimes_Over_Time.png"
    plt.figure(figsize=(10, 3))
    plt.step(applied.index, applied.values, where="post")
    plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
    plt.title("Applied Volatility Regimes Over Time (Weekly)")
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()
    return fp


def _plot_perf_panels(perf: pd.DataFrame, cfg: PipelineConfig, outdir: Path) -> Dict[str, Path]:
    figs: Dict[str, Path] = {}

    fp = outdir / "Figure_4.5_Cumulative_Returns.png"
    plt.figure(figsize=(10, 4))
    for c in perf.columns:
        v = equity_curve_from_log_returns(perf[c])
        plt.plot(v.index, v.values, label=c)
    plt.title("Cumulative Returns (Weekly)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()
    figs["cumulative"] = fp

    fp = outdir / "Figure_4.6_Drawdown.png"
    plt.figure(figsize=(10, 4))
    for c in perf.columns:
        dd = drawdown_from_equity_curve(equity_curve_from_log_returns(perf[c]))
        plt.plot(dd.index, dd.values, label=c)
    plt.title("Drawdown (Weekly)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()
    figs["drawdown"] = fp

    fp = outdir / "Figure_4.7_Rolling_Sharpe.png"
    plt.figure(figsize=(10, 4))
    for c in perf.columns:
        rs = rolling_sharpe_weekly(perf[c].dropna(), window_weeks=26, rf_annual=cfg.rf_annual)
        plt.plot(rs.index, rs.values, label=c)
    plt.title("Rolling Sharpe (26-week)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()
    figs["rolling_sharpe"] = fp

    return figs


def _plot_weights_heatmap(wdf: pd.DataFrame, outdir: Path) -> Path:
    w = wdf.drop(columns=["Policy"], errors="ignore")
    fp = outdir / "Figure_4.8_Weights_Heatmap_Weekly.png"
    if w.empty:
        return fp
    plt.figure(figsize=(12, 4))
    plt.imshow(w.T.values, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(w.columns)), w.columns)
    step = max(1, len(w.index) // 10)
    xticks = range(0, len(w.index), step)
    xlabels = [str(w.index[i].date()) for i in xticks]
    plt.xticks(list(xticks), xlabels, rotation=45, ha="right")
    plt.title("Weights Heatmap (Weekly)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fp, dpi=220)
    plt.close()
    return fp


def run_pipeline(xlsx_path: Path, outdir: Path, cfg: Optional[PipelineConfig] = None, make_zip: bool = True, use_lstm: bool = False) -> PipelineOutputs:
    cfg = cfg or PipelineConfig()
    outdir = _ensure_dir(Path(outdir))

    tables: Dict[str, Path] = {}
    figures: Dict[str, Path] = {}
    artifacts: Dict[str, Path] = {}

    prices = load_multisheet_prices(Path(xlsx_path), cfg)
    rets_d = to_log_returns(prices)

    figures["risk_return"] = _plot_risk_return_scatter(rets_d, outdir)
    figures["corr_heatmap"] = _plot_corr_heatmap(rets_d, outdir)

    feats = compute_weekly_features(rets_d, cfg.rebalance_freq)
    y_rule, (t1, t2) = rule_based_regimes(feats["mkt_vol_weekly"], cfg.regime_q_low, cfg.regime_q_high)

    pred = None
    cm = None
    metrics = {"accuracy": np.nan, "macro_f1": np.nan}

    if use_lstm:
        pred, cm, metrics = lstm_regime_classifier(feats, y_rule, cfg)

    if pred is None:
        pred = y_rule.rename("pred_regime")

    tables["regime_thresholds"] = outdir / "Table_Regime_Thresholds.csv"
    pd.DataFrame([{"t_low": t1, "t_high": t2}]).to_csv(tables["regime_thresholds"], index=False)

    tables["regime_series"] = outdir / "regime_series.csv"
    pred.to_csv(tables["regime_series"])

    if cm is not None:
        figures["confusion"] = _plot_confusion_matrix(cm, outdir)
        tables["classifier_metrics"] = outdir / "Table_Regime_Classifier_Metrics.csv"
        pd.DataFrame([metrics]).to_csv(tables["classifier_metrics"], index=False)

    # Policies: 0=MV, 1=RP, 2=MinVar
    policy_regime = pred.rename("policy_code")
    policy_mv = pd.Series(0, index=policy_regime.index, name="policy_code")
    policy_rp = pd.Series(1, index=policy_regime.index, name="policy_code")
    policy_min = pd.Series(2, index=policy_regime.index, name="policy_code")

    port_reg, w_reg, applied = backtest(rets_d, policy_regime, cfg)
    port_mv, _, _ = backtest(rets_d, policy_mv, cfg)
    port_rp, _, _ = backtest(rets_d, policy_rp, cfg)
    port_min, _, _ = backtest(rets_d, policy_min, cfg)

    tables["applied_regime"] = outdir / "applied_regime_by_rebalance.csv"
    applied.to_csv(tables["applied_regime"])
    figures["regimes"] = _plot_regimes(applied, outdir)

    perf = pd.concat(
        {
            "Regime-Conditioned": port_reg,
            "Static Mean-Variance": port_mv,
            "Static Risk-Parity": port_rp,
            "Static Minimum-Variance": port_min,
        },
        axis=1,
    ).dropna(how="all")

    tables["weekly_returns"] = outdir / "weekly_portfolio_log_returns.csv"
    perf.to_csv(tables["weekly_returns"])

    figs = _plot_perf_panels(perf, cfg, outdir)
    figures.update({f"perf_{k}": v for k, v in figs.items()})

    tables["weights_regime"] = outdir / "regime_conditioned_weights.csv"
    w_reg.to_csv(tables["weights_regime"])
    figures["weights_heatmap"] = _plot_weights_heatmap(w_reg, outdir)

    rows = []
    for name in perf.columns:
        rows.append({"Strategy": name, **perf_metrics_weekly(perf[name], rf_annual=cfg.rf_annual)})
    tables["performance"] = outdir / "Table_Portfolio_Performance_Summary.csv"
    pd.DataFrame(rows).to_csv(tables["performance"], index=False)

    meta = {
        "python": ">=3.9",
        "cvxpy_available": str(HAS_CVXPY),
        "tensorflow_available": str(HAS_TF),
        "use_lstm": str(bool(use_lstm)),
        "tc_bps": str(cfg.tc_bps),
        "rebalance_freq": cfg.rebalance_freq,
        "estimation_window_weeks": str(cfg.estimation_window_weeks),
        "cov_shrink_lam": str(cfg.cov_shrink_lam),
        "weight_cap": str(cfg.weight_cap),
        "regime_thresholds": f"{t1:.6g},{t2:.6g}",
    }
    artifacts["meta"] = outdir / "run_meta.csv"
    pd.DataFrame([meta]).to_csv(artifacts["meta"], index=False)

    if make_zip:
        zip_path = outdir.parent / "outputs_bundle.zip"
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.glob("*")):
                if fp.is_file():
                    z.write(fp, arcname=f"outputs/{fp.name}")
        artifacts["zip"] = zip_path

    return PipelineOutputs(outdir=outdir, tables=tables, figures=figures, artifacts=artifacts, meta=meta)
