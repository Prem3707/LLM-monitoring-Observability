"""
Streamlit Observability Dashboard.
Real-time view of latency, cost, error rate, and quality trends.

Run with: streamlit run src/dashboard/app.py
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np


DB_PATH = Path("data/observability.db")

st.set_page_config(
    page_title="LLM Observability Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 LLM Observability Dashboard")
st.caption("Real-time production metrics for your RAG system")


@st.cache_data(ttl=10)
def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM request_traces ORDER BY timestamp DESC LIMIT 1000", conn)
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_data()

if df.empty:
    st.warning("No traces yet. Run the RAG system to generate data.")
    st.stop()

# ---- KPI Row ----
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Requests", len(df))
col2.metric("p50 Latency", f"{np.percentile(df['latency_ms'].dropna(), 50):.0f}ms")
col3.metric("p95 Latency", f"{np.percentile(df['latency_ms'].dropna(), 95):.0f}ms")
col4.metric("Total Cost", f"${df['cost_usd'].sum():.4f}")
err_rate = df["error"].notna().mean() * 100
col5.metric("Error Rate", f"{err_rate:.1f}%")

st.divider()

# ---- Latency over time ----
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Latency Over Time")
    fig = px.scatter(
        df.sort_values("timestamp"),
        x="timestamp", y="latency_ms",
        color="model", opacity=0.7,
        labels={"latency_ms": "Latency (ms)"},
    )
    fig.add_hline(y=np.percentile(df["latency_ms"].dropna(), 95),
                  line_dash="dash", line_color="red",
                  annotation_text="p95", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Cost Per Request Over Time")
    df_sorted = df.sort_values("timestamp")
    df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
    fig2 = px.line(df_sorted, x="timestamp", y="cumulative_cost",
                   labels={"cumulative_cost": "Cumulative Cost (USD)"})
    st.plotly_chart(fig2, use_container_width=True)

# ---- Quality + Breakdown ----
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Latency Breakdown: Retrieval vs Generation")
    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=df["retrieval_latency_ms"], name="Retrieval", marker_color="#4e9af1"))
    fig3.add_trace(go.Box(y=df["generation_latency_ms"], name="Generation", marker_color="#f18f4e"))
    fig3.update_layout(yaxis_title="Latency (ms)")
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Quality Score Distribution")
    q_df = df["quality_score"].dropna()
    if not q_df.empty:
        fig4 = px.histogram(q_df, nbins=20, labels={"value": "Quality Score"})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No quality scores logged yet.")

# ---- Recent Traces Table ----
st.subheader("Recent Traces")
display_cols = ["trace_id", "timestamp", "model", "latency_ms", "cost_usd", "quality_score", "error"]
st.dataframe(
    df[display_cols].head(50).style.format({"cost_usd": "${:.6f}", "latency_ms": "{:.0f}ms"}),
    use_container_width=True,
)

st.caption("Refreshes every 10 seconds | Data from local SQLite store")
