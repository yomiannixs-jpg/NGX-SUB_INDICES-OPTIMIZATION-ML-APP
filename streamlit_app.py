
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ngx_portfolio.core import run_pipeline, PipelineConfig

st.set_page_config(page_title="NGX Portfolio Pipeline", layout="wide")
st.title("NGX Portfolio Pipeline")

uploaded = st.file_uploader("Upload NGX multi-sheet Excel workbook (.xlsx)", type=["xlsx"])

with st.sidebar:
    st.subheader("Run options")
    use_lstm = st.checkbox("Use LSTM regimes (requires tensorflow)", value=False)
    tc_bps = st.slider("Transaction costs (bps × turnover)", 0.0, 50.0, 10.0, 0.5)
    make_zip = st.checkbox("Create ZIP bundle", value=True)
    run = st.button("Run", type="primary", disabled=(uploaded is None))

if run and uploaded is not None:
    tmp = Path(tempfile.mkdtemp(prefix="ngx_"))
    xlsx = tmp / uploaded.name
    xlsx.write_bytes(uploaded.getbuffer())

    outdir = tmp / "outputs"
    cfg = PipelineConfig(tc_bps=float(tc_bps))

    with st.spinner("Running pipeline…"):
        outputs = run_pipeline(xlsx, outdir, cfg=cfg, make_zip=bool(make_zip), use_lstm=bool(use_lstm))

    st.success("Complete.")

    if "zip" in outputs.artifacts and outputs.artifacts["zip"].exists():
        st.download_button(
            "Download outputs_bundle.zip",
            data=outputs.artifacts["zip"].read_bytes(),
            file_name="outputs_bundle.zip",
            mime="application/zip",
        )

    st.header("Figures")
    for name, fp in outputs.figures.items():
        if fp.exists():
            st.subheader(name)
            st.image(str(fp), use_container_width=True)

    st.header("Tables")
    for name, fp in outputs.tables.items():
        if fp.exists():
            st.subheader(name)
            st.dataframe(pd.read_csv(fp), use_container_width=True)
