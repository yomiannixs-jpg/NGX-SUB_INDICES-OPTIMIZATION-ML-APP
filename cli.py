
from __future__ import annotations

import argparse
from pathlib import Path

from ngx_portfolio.core import run_pipeline, PipelineConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Run NGX portfolio pipeline on a multi-sheet Excel workbook.")
    p.add_argument("--xlsx", required=True, help="Path to Excel workbook (.xlsx)")
    p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--no-zip", action="store_true", help="Do not create outputs_bundle.zip")
    p.add_argument("--use-lstm", action="store_true", help="Use TensorFlow LSTM regime classifier (requires tensorflow)")
    p.add_argument("--tc-bps", type=float, default=10.0, help="Transaction costs (bps × turnover), default 10")
    args = p.parse_args()

    cfg = PipelineConfig(tc_bps=float(args.tc_bps))
    run_pipeline(Path(args.xlsx), Path(args.outdir), cfg=cfg, make_zip=not args.no_zip, use_lstm=bool(args.use_lstm))
    print(f"Done. Outputs written to: {Path(args.outdir).resolve()}")
    return 0
