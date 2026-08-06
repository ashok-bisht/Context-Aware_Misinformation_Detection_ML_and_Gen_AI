"""
Lightweight RAG: retrieves the most relevant rows from the fact-check
knowledge base (stored in S3, seeded from a local default on first run)
using TF-IDF cosine similarity, then asks a free Groq-hosted model to
explain the model's prediction, grounded in that context plus the
sentiment signal from utils/sentiment.py.

Swap `load_knowledge_base` for a vector DB (Chroma/FAISS/pgvector) later
without touching app.py -- it only calls these four functions.
"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.s3_storage import read_csv_from_s3, seed_s3_csv_if_missing


@st.cache_resource(show_spinner=False)
def load_knowledge_base(bucket: str, kb_key: str, aws_region: str, local_fallback: str = "data/fact_checks_sample.csv"):
    seed_s3_csv_if_missing(bucket, kb_key, local_fallback, aws_region)
    df = read_csv_from_s3(bucket, kb_key, aws_region)
    if df is None:
        df = pd.read_csv(local_fallback)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["text"].fillna(""))
    return df, vectorizer, matrix


def retrieve_context(query: str, df, vectorizer, matrix, top_k: int = 3):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    results = df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results[results["similarity"] > 0]


def build_explanation_prompt(claim: str, prediction_label: str, confidence: float, retrieved_df, sentiment: dict) -> str:
    if retrieved_df.empty:
        context_block = "No closely related fact-checks were found in the knowledge base."
    else:
        context_block = "\n\n".join(
            f"- Source: {row['source']}\n  Verdict: {row['verdict']}\n  Snippet: {row['text']}"
            for _, row in retrieved_df.iterrows()
        )

    return f"""You are a media-literacy assistant explaining why a machine learning model classified a claim as it did.

Claim submitted by the user:
"{claim}"

Model prediction: {prediction_label} (confidence: {confidence:.1%})

Independent sentiment analysis of the text: {sentiment['label']} (compound score: {sentiment['compound']:.2f},
positive: {sentiment['pos']:.2f}, negative: {sentiment['neg']:.2f}, neutral: {sentiment['neu']:.2f})

Relevant fact-check context retrieved from the knowledge base:
{context_block}

Write a concise, neutral explanation (3-5 sentences) of why this claim was likely flagged as {prediction_label}.
Ground your reasoning in the retrieved context where it's relevant, and mention the sentiment reading only if
it's actually informative (e.g. strongly negative/inflammatory tone is a common but not definitive marker of
low-credibility content). Say plainly if the context is insufficient to fully confirm the model's prediction.
Don't state anything as verified fact unless it's supported by the retrieved context above."""


def get_llm_explanation(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
