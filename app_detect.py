import pandas as pd
import plotly.express as px
import streamlit as st

from utils.dashboard import log_prediction, render_dashboard
from utils.model_loader import load_models
from utils.rag_engine import build_explanation_prompt, get_llm_explanation, load_knowledge_base, retrieve_context
from utils.sentiment import analyze_sentiment

st.set_page_config(page_title="Fake News Detector", page_icon="\U0001f50e", layout="wide")

st.title("\U0001f50e Fake News Detector")
st.caption("AWS S3 \u00b7 XGBoost (TF-IDF) \u00b7 sentiment Analysis \u00b7 Groq (free LLM) \u00b7 Dashboard")

with st.sidebar:
    st.header("S3 Configuration")
    bucket = st.text_input("S3 bucket", value=st.secrets.get("S3_BUCKET", "context-aware-misinformation-detection"))
    aws_region = st.text_input("AWS region", value=st.secrets.get("AWS_REGION", "us-east-1"))
    vectorizer_key = st.text_input("Vectorizer key", value="pickle_file/tfidf_vectorizer.pkl") 
    model_key = st.text_input("Model key", value="pickle_file/XGBoost.pkl")
    kb_key = st.text_input("Knowledge base key", value="knowledge-base/fact_checks.csv")
    log_key = st.text_input("Prediction log key", value="logs/predictions_log.csv")

    st.divider()
    st.header("Free LLM (Groq)")
    groq_key = st.text_input("Groq API key", type="password", value=st.secrets.get("GROQ_API_KEY", ""))
    st.caption("Free key at console.groq.com \u2014 no credit card required.")
    llm_model = st.text_input("Groq model", value="llama-3.1-8b-instant")

    #st.divider()
    #st.header("Tableau (optional)")
    #tableau_url = st.text_input("Tableau Public embed URL", value=st.secrets.get("TABLEAU_URL", ""))
    #st.caption("Leave blank to use the built-in charts instead. Tableau Public is free but published views are publicly visible.")

tab1, tab2, tab3 = st.tabs(["\U0001f575\ufe0f Detect", "\U0001f9e0 Explain (GenAI + RAG)", "\U0001f4ca Dashboard"])

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_sentiment" not in st.session_state:
    st.session_state.last_sentiment = None

with tab1:
    st.subheader("Paste a claim or article excerpt")
    claim_text = st.text_area("Text to analyze", height=200, key="claim_input")

    if st.button("Analyze", type="primary"):
        if not claim_text.strip():
            st.warning("Please paste some text first.")
        else:
            try:
                vectorizer, model = load_models(bucket, vectorizer_key, model_key, aws_region)
                X = vectorizer.transform([claim_text])

                # Works for binary (TRUE/FALSE) or the full 6-class LIAR labels --
                # the XGBoost classifier stores whatever label strings it was
                # trained on in model.classes_, so no hardcoded 0/1 mapping is needed.
                pred_label = str(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                classes = [str(c) for c in model.classes_]
                confidence = float(proba[classes.index(pred_label)])
                label = pred_label.upper()
            
                # Rough color grouping across binary or 6-class LIAR labels
                lbl_lower = pred_label.lower()
                if lbl_lower in ("true", "mostly-true"):
                    icon = "\U0001f7e2"
                elif lbl_lower in ("half-true", "barely-true"):
                    icon = "\U0001f7e1"
                else:  # false, pants-fire
                    icon = "\U0001f534"
            
                sentiment = analyze_sentiment(claim_text)
            
                st.session_state.last_prediction = {"claim": claim_text, "label": label, "confidence": confidence}
                st.session_state.last_sentiment = sentiment
            
                log_prediction(bucket, log_key, aws_region, claim_text, label, confidence, sentiment["label"])
            
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {icon} {label}")
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.1%}")
                with col2:
                    st.markdown(f"### Sentiment: {sentiment['label']}")
                    st.caption(f"Compound score: {sentiment['compound']:.2f}")
            
                if len(classes) > 2:
                    with st.expander("Full probability breakdown (all classes)"):
                        prob_df = pd.DataFrame({"label": classes, "probability": proba}).sort_values("probability", ascending=False)
                        fig = px.bar(prob_df, x="label", y="probability", title="Class Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.info("Double-check the S3 bucket/key names and that AWS credentials are available to the app.")

with tab2:
    st.subheader("Why did the model decide this?")
    if not st.session_state.last_prediction:
        st.info("Run an analysis in the Detect tab first.")
    elif not groq_key:
        st.warning("Add your free Groq API key in the sidebar to generate an explanation.")
    else:
        if st.button("Generate Explanation"):
            with st.spinner("Retrieving context and generating explanation..."):
                try:
                    df, kb_vectorizer, kb_matrix = load_knowledge_base(bucket, kb_key, aws_region)
                    retrieved = retrieve_context(st.session_state.last_prediction["claim"], df, kb_vectorizer, kb_matrix)
                    prompt = build_explanation_prompt(
                        st.session_state.last_prediction["claim"],
                        st.session_state.last_prediction["label"],
                        st.session_state.last_prediction["confidence"],
                        retrieved,
                        st.session_state.last_sentiment,
                    )
                    explanation = get_llm_explanation(prompt, groq_key, llm_model)
                    st.markdown(explanation)

                    if not retrieved.empty:
                        with st.expander("Retrieved context used"):
                            st.dataframe(retrieved[["source", "verdict", "text", "similarity"]], use_container_width=True)
                except Exception as e:
                    st.error(f"Explanation generation failed: {e}")

#with tab3:
 #   st.subheader("Prediction Analytics")
 #   render_dashboard(bucket, log_key, aws_region, tableau_url)

with tab3:
    st.subheader("Prediction Analytics")
    render_dashboard(bucket, log_key, aws_region)