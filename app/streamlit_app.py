import streamlit as st
import joblib
import os
import json
from sentence_transformers import SentenceTransformer

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# load
model = joblib.load(MODEL_PATH)
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# =========================
# UI
# =========================
st.set_page_config(page_title="Türkçe Sarkazm AI", page_icon="😏")

st.title("😏 Türkçe Sarkazm vs Normal Cümle Analizi")

st.subheader("📊 Model Performansı")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
col2.metric("Precision", f"{metrics['precision']:.2f}")
col3.metric("Recall", f"{metrics['recall']:.2f}")
col4.metric("F1 Score", f"{metrics['f1']:.2f}")

st.caption("ML + RAG destekli sınıflandırma sistemi")

text = st.text_area("Cümleyi gir:")

col1, col2 = st.columns(2)

with col1:
    run = st.button("Tahmin Et")

with col2:
    st.button("Temizle", on_click=lambda: st.rerun())

# =========================
# PREDICTION
# =========================
if run:
    if not text.strip():
        st.warning("Boş bırakma.")
    else:
        # Embedding üret
        X = embedding_model.encode([text])
        pred = model.predict(X)[0]

        # probability varsa
        try:
            proba = model.predict_proba(X)[0]
            sarcasm_score = proba[1]
        except:
            sarcasm_score = None

        # result
        if pred == 1:
            st.error("😏 SARKASTİK")
        else:
            st.success("🙂 NORMAL")

        # score
        if sarcasm_score is not None:
            st.progress(float(sarcasm_score))
            st.write(f"Sarkazm skoru: %{sarcasm_score*100:.2f}")
