# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import random
import textwrap
from typing import List, Tuple, Dict

# ML & NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Optional DistilBERT
HAS_TRANSFORMERS = False
try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

st.set_page_config(page_title="Customer Complaint Classification", layout="wide")

# -------------------------
# Synthetic complaint generator
# -------------------------
@st.cache_data
def generate_synthetic_complaints(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    categories = ["delayed", "lost", "damaged", "billing"]

    templates = {
        "delayed": [
            "My package hasn't arrived. It was due {days} days ago.",
            "Still waiting for my delivery. No tracking updates since last week.",
            "Parcel delayed — expected it earlier. Order #{order_id}.",
            "Delivery delay beyond promised window — unacceptable."
        ],
        "lost": [
            "Tracking says delivered but I got nothing. Package is lost.",
            "My parcel is missing — please investigate.",
            "Order never arrived. Likely lost in transit.",
            "Package lost. Delivery shows completed but I didn’t receive it."
        ],
        "damaged": [
            "My package arrived damaged and contents broken.",
            "Item was cracked on delivery — box was crushed.",
            "Received damaged goods. Replacement needed.",
            "Product arrived broken — attaching photos."
        ],
        "billing": [
            "I was overcharged — billing seems incorrect.",
            "Charged twice for same order. Refund needed.",
            "Wrong billing amount shown on invoice.",
            "Billing issue — unexpected fees added."
        ]
    }

    rows = []
    for i in range(n):
        cat = random.choice(categories)
        t = random.choice(templates[cat])
        days = random.randint(1, 10)
        order_id = random.randint(10000, 99999)
        text = t.format(days=days, order_id=order_id)

        # noise
        if random.random() < 0.12:
            text += " " + random.choice(["Please help.", "Need urgent resolution.", "Thanks."])
        if random.random() < 0.06:
            text = text.upper()

        rows.append({"text": text, "category": cat})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

# -------------------------
# Preprocessing
# -------------------------
def preprocess_text(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        t = str(t).replace("\n", " ").strip()
        t = " ".join(t.split())
        out.append(t)
    return out

# -------------------------
# Keyword extraction
# -------------------------
def top_keywords(vectorizer: TfidfVectorizer, clf: LogisticRegression, le: LabelEncoder, top_n: int = 10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_
    labels = le.inverse_transform(np.arange(len(coefs)))

    result = {}
    for idx, lab in enumerate(labels):
        row = coefs[idx]
        top_idx = np.argsort(row)[-top_n:][::-1]
        result[lab] = [(feature_names[i], float(row[i])) for i in top_idx]
    return result

# -------------------------
# TF-IDF + Logistic Regression Training
# -------------------------
@st.cache_data
def train_model(df: pd.DataFrame, test_size=0.2, seed=42):
    texts = preprocess_text(df["text"].tolist())
    labels = df["category"].tolist()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=seed, stratify=y
    )

    vect = TfidfVectorizer(ngram_range=(1,2), max_features=10000, stop_words="english")
    clf = LogisticRegression(max_iter=300, solver="liblinear")
    pipe = make_pipeline(vect, clf)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    kw = top_keywords(vect, clf, le, top_n=10)

    return {
        "pipeline": pipe,
        "le": le,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "cm": cm,
        "report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
        "keywords": kw
    }

# -------------------------
# Optional DistilBERT
# -------------------------
@st.cache_resource
def load_distilbert():
    if not HAS_TRANSFORMERS:
        return None

    try:
        return hf_pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
    except:
        return None

def bert_predict(pipe, text):
    out = pipe(text)[0]
    best = max(out, key=lambda x: x["score"])
    return best["label"], best["score"]

# ----------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------

st.title("📬 Customer Complaint Classification")
st.markdown("Automatically classify complaints into **delayed, lost, damaged, billing**.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    n = st.number_input("Synthetic dataset size", 1000, 4000, 1500, step=100)
    test_size = st.slider("Test split", 0.1, 0.4, 0.2)
    seed = st.number_input("Random Seed", 1, 9999, 42)
    st.markdown("---")
    st.subheader("DistilBERT (Optional)")
    use_bert = st.checkbox("Enable DistilBERT demo")

# Data Load
st.subheader("Dataset")
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if "text" not in df:
        st.error("CSV must contain a 'text' column.")
        st.stop()
    if "category" not in df:
        st.warning("No 'category' column found — using synthetic labels.")
        df_syn = generate_synthetic_complaints(n, seed)
        df["category"] = df_syn["category"].iloc[:len(df)]
else:
    df = generate_synthetic_complaints(n, seed)

st.write(df.sample(8))

# Train Model
st.subheader("Train TF-IDF + Logistic Regression Model")

if st.button("Train Model"):
    with st.spinner("Training..."):
        art = train_model(df, test_size, seed)
    st.session_state["artifacts"] = art
    st.success("Training completed.")

    # Report
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(art["report"]).transpose())

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(art["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=art["le"].classes_,
                yticklabels=art["le"].classes_,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # Top Keywords
    st.write("### 🔑 Top Keywords per Category")
    for cat, words in art["keywords"].items():
        st.markdown(f"**{cat}**: " + ", ".join([w for w,_ in words]))

# DistilBERT
if use_bert:
    if not HAS_TRANSFORMERS:
        st.error("transformers package not installed.")
    else:
        bert = load_distilbert()
        if bert:
            st.success("DistilBERT loaded successfully.")
        else:
            st.error("Unable to load DistilBERT model.")

# ------------------------
# Single Complaint Prediction
# ------------------------
st.subheader("Classify a Complaint")

text_input = st.text_area("Paste complaint text here:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict using TF-IDF + LR"):
        if "artifacts" not in st.session_state:
            st.error("Train the model first.")
        else:
            art = st.session_state["artifacts"]
            vec = art["pipeline"]
            le = art["le"]
            p = vec.predict([text_input])[0]
            pred = le.inverse_transform([p])[0]

            st.success(f"Predicted Category: **{pred}**")

with col2:
    if st.button("Predict using DistilBERT"):
        if not use_bert:
            st.error("Enable DistilBERT in sidebar.")
        else:
            p = load_distilbert()
            if p:
                label, score = bert_predict(p, text_input)
                st.info(f"DistilBERT Label: **{label}** (Score: {score:.2f})")

st.markdown("---")
st.caption("NLP + ML pipeline with synthetic dataset. Ideal for portfolio, logistics, and customer service analytics use cases.")
