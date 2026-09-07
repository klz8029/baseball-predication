"""Live calibration & P&L dashboard, read straight from the DuckDB warehouse.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warehouse.db import DB_PATH, build, connect  # noqa: E402

st.set_page_config(page_title="Baseball win-prob model", layout="wide")


@st.cache_data(ttl=300)
def load():
    if not os.path.exists(DB_PATH):
        build(verbose=False)
    with connect(read_only=True) as con:
        return {t: con.sql(f"SELECT * FROM {t}").df()
                for t in ("mart_performance", "mart_calibration", "mart_daily")}


d = load()
perf, calib, daily = d["mart_performance"], d["mart_calibration"], d["mart_daily"]

st.title("MLB win-probability model")
st.caption("Monte-Carlo game simulator vs. the Kalshi prediction market. "
           "Validated walk-forward, then forward-tested live.")

fw = perf.set_index("source").reindex(["walkforward", "forward", "market"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Walk-forward Brier", f"{fw.loc['walkforward','brier']:.4f}",
          f"{fw.loc['walkforward','brier'] - fw.loc['market','brier']:+.4f} vs market")
c2.metric("Walk-forward ROI", f"{fw.loc['walkforward','roi']*100:+.1f}%",
          f"{int(fw.loc['walkforward','n_bets'])} bets")
c3.metric("Live Brier", f"{fw.loc['forward','brier']:.4f}",
          f"{fw.loc['forward','brier'] - fw.loc['market','brier']:+.4f} vs market")
live_roi = daily["cum_pnl"].iloc[-1] / max(daily["staked"].sum(), 1e-9) if len(daily) else 0
c4.metric("Live P&L", f"${daily['cum_pnl'].iloc[-1]:+.2f}" if len(daily) else "—",
          f"{int(daily['cum_bets'].iloc[-1])} bets" if len(daily) else "")

st.info("**Verdict: no live edge.** The walk-forward +12.6% ROI was a favourable "
        "two-month stretch. The model only matches the market on Brier, and the "
        "live closing-line-value beat rate has stayed near 40% — the market moves "
        "against the model's picks after it bets.")

left, right = st.columns(2)

with left:
    st.subheader("Calibration")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#888"),
                    name="perfect", hoverinfo="skip")
    for src, color in [("walkforward", "#B4481F"), ("forward", "#3B6B86"), ("market", "#3C7A4E")]:
        s = calib[calib.source == src].sort_values("mean_pred")
        if len(s):
            fig.add_scatter(x=s.mean_pred, y=s.actual_rate, mode="lines+markers",
                            name=src, line=dict(color=color),
                            marker=dict(size=6 + 18 * s.n / s.n.max()))
    fig.update_layout(xaxis_title="predicted P(home win)", yaxis_title="observed rate",
                      height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Live P&L and closing-line value")
    if len(daily):
        fig = go.Figure()
        fig.add_scatter(x=daily.game_date, y=daily.cum_pnl, mode="lines",
                        name="cumulative P&L ($)", line=dict(color="#B4481F", width=2))
        fig.add_bar(x=daily.game_date, y=daily.pnl, name="daily P&L",
                    marker_color="#c9c9c9", opacity=0.6)
        fig.add_scatter(x=daily.game_date, y=daily.beat_close_last30, name="beat-close rate (30)",
                        yaxis="y2", line=dict(color="#3B6B86", dash="dash"))
        fig.add_hline(y=0.5, line=dict(color="#888", dash="dot"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(title="P&L ($)"),
                          yaxis2=dict(title="beat-close rate", overlaying="y", side="right",
                                      range=[0, 1]),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("no settled forward bets yet")

st.subheader("Scoreboard")
st.dataframe(perf, use_container_width=True, hide_index=True)

st.subheader("Model-config experiment log")
st.caption("Every combiner tried and rejected — full history in MLflow "
           "(`mlflow ui --backend-store-uri sqlite:///mlflow.db`).")
st.code("python experiments/track.py   # (re)log the sweep\nmlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")
