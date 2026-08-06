"""
S3 is used for three things in this app, so nothing depends on local disk
(which doesn't persist on most free-tier hosts):

1. models-pkl-file/  -> the trained vectorizer + XGBoost model (already there)
2. knowledge-base/   -> the RAG fact-check corpus (seeded from a local default
                         the first time the app runs, then lives in S3)
3. logs/              -> the prediction log that powers the Dashboard tab
"""

import io

import boto3
import pandas as pd
import streamlit as st


def _get_client(aws_region: str):
    kwargs = {"region_name": aws_region}
    if "AWS_ACCESS_KEY_ID" in st.secrets and "AWS_SECRET_ACCESS_KEY" in st.secrets:
        kwargs["aws_access_key_id"] = st.secrets["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    return boto3.client("s3", **kwargs)


def read_csv_from_s3(bucket: str, key: str, aws_region: str):
    s3 = _get_client(aws_region)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return None


def append_row_to_s3_csv(bucket: str, key: str, row: dict, aws_region: str) -> None:
    existing = read_csv_from_s3(bucket, key, aws_region)
    new_row = pd.DataFrame([row])
    df = pd.concat([existing, new_row], ignore_index=True) if existing is not None else new_row
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    _get_client(aws_region).put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def seed_s3_csv_if_missing(bucket: str, key: str, local_default_path: str, aws_region: str) -> None:
    """First run only: uploads the bundled default CSV to S3 so the knowledge
    base lives centrally afterward and survives redeploys."""
    s3 = _get_client(aws_region)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        with open(local_default_path, "rb") as f:
            s3.put_object(Bucket=bucket, Key=key, Body=f.read())
