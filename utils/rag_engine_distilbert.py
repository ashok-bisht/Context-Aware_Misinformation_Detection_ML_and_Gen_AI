"""
DistilBERT-based retrieval for the RAG "Explain" tab.

Drop-in replacement for the TF-IDF versions of `load_knowledge_base` and
`retrieve_context` in utils/rag_engine.py. Keeps the same call signatures
used in app.py:

    df, kb_vectorizer, kb_matrix = load_knowledge_base(bucket, kb_key, aws_region)
    retrieved = retrieve_context(claim, df, kb_vectorizer, kb_matrix)

`kb_vectorizer` is now the SentenceTransformer model (instead of a fitted
TfidfVectorizer), and `kb_matrix` is a numpy array of embeddings (instead of
a sparse TF-IDF matrix). build_explanation_prompt / get_llm_explanation are
untouched -- they only consume the resulting DataFrame.
"""

import io

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# multi-qa-distilbert-cos-v1 is a DistilBERT model fine-tuned specifically
# for semantic search / cosine-similarity retrieval (question <-> passage),
# which is a better fit for RAG-style lookups than generic DistilBERT.
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-distilbert-cos-v1"


@st.cache_resource(show_spinner=False)
def get_embedder():
    """Load and cache the DistilBERT sentence-embedding model once per session."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data(show_spinner="Loading knowledge base and building embeddings...")
def load_knowledge_base(bucket: str, kb_key: str, aws_region: str):
    """
    Download the fact-check CSV from S3 and embed every row's text with
    DistilBERT. Cached on (bucket, kb_key, aws_region) so re-runs don't
    re-embed the whole knowledge base every time.

    Expects the CSV to have at least: source, verdict, text
    """
    s3 = boto3.client("s3", region_name=aws_region)
    obj = s3.get_object(Bucket=bucket, Key=kb_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    embedder = get_embedder()
    embeddings = embedder.encode(
        df["text"].astype(str).tolist(),
        normalize_embeddings=True,   # so cosine similarity == dot product
        show_progress_bar=False,
    )

    return df, embedder, np.asarray(embeddings)


def retrieve_context(claim: str, df: pd.DataFrame, embedder, kb_matrix: np.ndarray, top_k: int = 3, min_similarity: float = 0.3) -> pd.DataFrame:
    """
    Embed the claim, rank the knowledge base by cosine similarity, and
    return the top_k matches above min_similarity as a DataFrame with a
    `similarity` column (same shape the app.py expects).
    """
    if df.empty or kb_matrix.size == 0:
        return df.head(0).assign(similarity=[])

    claim_vec = embedder.encode([claim], normalize_embeddings=True, show_progress_bar=False)[0]

    # embeddings are normalized, so cosine similarity is just the dot product
    similarities = kb_matrix @ claim_vec

    ranked_idx = np.argsort(-similarities)[:top_k]
    results = df.iloc[ranked_idx].copy()
    results["similarity"] = similarities[ranked_idx]

    results = results[results["similarity"] >= min_similarity]
    return results.reset_index(drop=True)
