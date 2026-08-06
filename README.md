<<<<<<< HEAD
# Fake News Detector — Streamlit App (100% Free Tier)

Three tabs, all backed by S3 so nothing depends on local disk:

1. **Detect** — pulls `tfidf_vectorizer.pkl` and `naive_bayes_model.pkl` from S3,
   vectorizes the pasted text, and classifies it. Works with either a binary
   TRUE/FALSE model or the full 6-class LIAR labels (true / mostly-true /
   half-true / barely-true / false / pants-fire) — whatever the model was
   trained on, since it reads the label set from `model.classes_`.
2. **Explain (GenAI + RAG)** — retrieves the closest matches from a fact-check
   knowledge base (stored in S3, seeded automatically from a starter CSV on first
   run) via TF-IDF similarity, runs local VADER sentiment analysis, and asks a
   **free Groq-hosted Llama model** to explain the prediction grounded in both.
3. **Dashboard** — an optional embedded **Tableau Public** view (free to publish
   to), plus native Plotly charts built from a prediction log stored in S3.

## Why everything is free

| Piece | Free option used | Notes |
|---|---|---|
| Model hosting | S3 (existing bucket) | Free tier: 5GB storage, 20k GET/mo for 12 months |
| Prediction log storage | S3 | Same bucket, new `logs/` prefix |
| RAG knowledge base | S3 | Same bucket, new `knowledge-base/` prefix |
| Sentiment analysis | NLTK VADER | Fully local, no API, no cost, downloads a ~130KB lexicon once |
| LLM explanation | Groq | Free API key at console.groq.com, no credit card |
| Dashboard | Tableau Public (optional) | Free to publish, **but published views are public to anyone with the link** |
| App hosting | Streamlit Community Cloud (suggested) | Free tier; this is exactly why logs/KB live in S3 instead of local disk |

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in AWS creds + your free Groq key
streamlit run app.py
```

Get a free Groq key: https://console.groq.com/keys — takes under a minute, no
credit card. All config (bucket, keys, model name, Tableau URL) is also editable
live in the sidebar.

## S3 layout this app expects

```
your-bucket/
├── models-pkl-file/
│   ├── tfidf_vectorizer.pkl        <- TfidfVectorizer fit on 'statement' ONLY
│   └── naive_bayes_model.pkl       <- MultinomialNB fit on the TF-IDF matrix
├── knowledge-base/
│   └── fact_checks.csv        <- auto-created on first run if missing
└── logs/
    └── predictions_log.csv    <- auto-created as you run predictions
```

## Growing the RAG knowledge base

Edit `knowledge-base/fact_checks.csv` directly in S3 (or re-upload a new version)
with columns: `text`, `source`, `verdict`. No redeploy needed — the app reads it
fresh each session.

## Adding a Tableau dashboard

1. Build your dashboard in Tableau Desktop (free trial) or Tableau Public Desktop
   (free, always).
2. Publish to Tableau Public.
3. Copy the view's embed/share URL into the sidebar's "Tableau Public embed URL"
   field (or `TABLEAU_URL` in secrets.toml).
4. **Caution:** Tableau Public dashboards are visible to anyone with the link —
   don't publish real user claims or PII to it. Consider feeding it aggregated
   stats exported from the S3 log instead of raw text.

## Notes on the model

- **Train on `statement` text only.** The original LIAR dataset also has
  `speaker`, `venue`, `party affiliation`, and — critically — five
  "credit history" count columns (`barely true counts`, `false counts`, etc).
  Those counts are each speaker's *historical tally of past labels*, so a
  model trained on them leaks the answer and will show suspiciously high
  accuracy (99%+) that has nothing to do with actually reading the text. A
  Streamlit user pasting a claim can't supply speaker history anyway — only
  the text-based `tfidf_vectorizer.pkl` + `naive_bayes_model.pkl` pair works
  for real predictions on new input.
- Use `MultinomialNB`, not `GaussianNB` — it's the right Naive Bayes variant
  for sparse TF-IDF/count features.
- Fit the vectorizer on the training split only, never on the full dataset
  (fitting on everything before splitting leaks test-set vocabulary into
  training and inflates the reported accuracy).
- Honest accuracy range for text-only LIAR classification: roughly 20-30% on
  the full 6-class problem, ~55-65% if collapsed to binary TRUE/FALSE. Treat
  anything near 99% as a bug to investigate, not a result to ship.
- The app reads whatever labels the model was actually trained on from
  `model.classes_` — no code changes needed to switch between binary and
  6-class.
- Groq model `llama-3.1-8b-instant` is fast and free-tier friendly; swap for a
  larger Groq-hosted model in the sidebar if you want stronger explanations at
  the cost of speed.
=======
# -Context-Aware_Misinformation_Detection_Using_ML_Gen_AI
 Context-Aware Misinformation Detection using ML and Gen AI Architecture
>>>>>>> 72ea8f2a9acb423cf575a26ac7ad7d780f5899c0
