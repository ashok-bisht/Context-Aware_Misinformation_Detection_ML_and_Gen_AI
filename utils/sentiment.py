"""
VADER sentiment analysis - fully local, free, no API key. Runs once to
download its small lexicon file (~130KB) then works offline. Used as a
second, independent signal alongside the XGBoost prediction: strongly
negative/inflammatory sentiment is a common (not definitive) marker of
low-credibility content, and it gives the RAG explanation something
concrete to reference.
"""

import nltk
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    scores = get_analyzer().polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    scores["label"] = label
    return scores
