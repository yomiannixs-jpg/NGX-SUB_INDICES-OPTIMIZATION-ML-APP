
# NGX Portfolio Pipeline (GitHub-ready, cross-platform)

✅ No auto-interpretation / executive summaries  
✅ Windows / macOS / Linux, Python 3.9+  
✅ CLI + Streamlit UI

## Install

Base:
```bash
pip install -e .
```

With CVXPY (recommended for constrained MV/min-var):
```bash
pip install -e ".[cvxpy]"
```

With TensorFlow LSTM regimes (research):
> TensorFlow requires **NumPy < 2.0**
```bash
pip install -e ".[lstm]"
```

## Run (CLI)
```bash
ngx-run --xlsx "NGX_Sector_Data original.xlsx" --outdir outputs
```

Optional:
```bash
ngx-run --xlsx data.xlsx --outdir outputs --use-lstm --tc-bps 10
```

## Run (Streamlit)
```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

## Input format
Excel workbook with multiple sheets. Each sheet should have:
- `Date`
- price column: one of `PX_LAST`, `Close`, `Price`, `Adj Close`
