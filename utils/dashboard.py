"""
Analytics tab. Two layers:

1. If you've published a Tableau Public view, its URL gets embedded via
   iframe at the top (Tableau Public hosting is free, but note: anything
   published there is publicly visible to anyone with the link).
2. Underneath (or instead, if no Tableau URL is set), native Plotly charts
   are built from the prediction log -- which lives in S3, not local disk,
   so it survives redeploys on free-tier hosts.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from utils.s3_storage import append_row_to_s3_csv, read_csv_from_s3


def log_prediction(bucket: str, log_key: str, aws_region: str, claim: str, label: str, confidence: float, sentiment_label: str) -> None:
    row = {
        "timestamp": pd.Timestamp.now(),
        "claim": claim[:200],
        "label": label,
        "confidence": confidence,
        "sentiment": sentiment_label,
    }
    append_row_to_s3_csv(bucket, log_key, row, aws_region)

def render_dashboard(bucket: str, log_key: str, aws_region: str) -> None:
#def render_dashboard(bucket: str, log_key: str, aws_region: str, tableau_url: str = "") -> None:
    #if tableau_url:
     #   st.caption("Embedded Tableau Public view")
      #  components.iframe(tableau_url, height=800, scrolling=True)
       # st.divider()
       # st.caption("Live prediction analytics (from S3 log)")

    df = read_csv_from_s3(bucket, log_key, aws_region)
    if df is None or df.empty:
        st.info("No predictions logged yet. Run a few checks in the Detect tab first.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col1, col2 = st.columns(2)
    with col1:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        fig = px.pie(counts, names="label", values="count", title="Prediction Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="confidence", color="label", nbins=20, title="Confidence Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "sentiment" in df.columns:
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            fig4 = px.bar(sent_counts, x="sentiment", y="count", title="Sentiment of Analyzed Claims")
            st.plotly_chart(fig4, use_container_width=True)
    with col4:
        df_sorted = df.sort_values("timestamp")
        fig3 = px.line(df_sorted, x="timestamp", y="confidence", color="label", markers=True, title="Predictions Over Time")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Recent Predictions")
    st.dataframe(df.sort_values("timestamp", ascending=False).head(20), use_container_width=True)
