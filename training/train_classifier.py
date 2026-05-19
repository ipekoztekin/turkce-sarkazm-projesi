import pandas as pd
import json
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Dataset oku
df = pd.read_csv("../data/processed_dataset.csv")

# Text ve label
texts = df["clean_text"].tolist()
labels = df["label"].tolist()

# Embedding modeli
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Embedding üret
embeddings = embedding_model.encode(texts)

# Train/Test ayır
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    labels,
    test_size=0.2,
    random_state=42
)

# Model oluştur
classifier = LogisticRegression()

# Eğit
classifier.fit(X_train, y_train)

# Tahmin
predictions = classifier.predict(X_test)

# Metrikler
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, predictions))

# Metrikleri kaydet
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1)
}

with open("../models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# MODELLERİ KAYDET
joblib.dump(classifier, "../models/model.pkl")
# SentenceTransformer pickle ile kaydedilmez, Streamlit tarafında yeniden yüklenmeli
