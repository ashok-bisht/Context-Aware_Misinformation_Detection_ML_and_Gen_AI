"""
Downloads and caches the TF-IDF vectorizer and XGBoost model from AWS S3.

Supports models saved using:
    joblib.dump(...)
"""

import os
import tempfile
import joblib
import boto3
import streamlit as st


@st.cache_resource(show_spinner="Loading models from S3...")
def load_models(
    bucket_name: str,
    vectorizer_key: str,
    model_key: str,
    aws_region: str = "us-east-1"
):

    session_kwargs = {
        "region_name": aws_region
    }

    # Load AWS credentials from Streamlit secrets if available
    if (
        "AWS_ACCESS_KEY_ID" in st.secrets
        and "AWS_SECRET_ACCESS_KEY" in st.secrets
    ):
        session_kwargs["aws_access_key_id"] = st.secrets["AWS_ACCESS_KEY_ID"]
        session_kwargs["aws_secret_access_key"] = st.secrets["AWS_SECRET_ACCESS_KEY"]

    s3 = boto3.client("s3", **session_kwargs)

    def download_and_load(s3_key):

        # Create a temporary local file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            temp_path = tmp.name

        # Download model from S3
        s3.download_file(
            bucket_name,
            s3_key,
            temp_path
        )

        # Debug information
        print(f"Downloaded: {s3_key}")
        print(f"Local file: {temp_path}")
        print(f"File size: {os.path.getsize(temp_path)} bytes")

        # Load using joblib
        obj = joblib.load(temp_path)

        # Remove temporary file
        os.remove(temp_path)

        return obj

    vectorizer = download_and_load(vectorizer_key)
    model = download_and_load(model_key)

    return vectorizer, model