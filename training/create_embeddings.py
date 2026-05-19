import pandas as pd

from sentence_transformers import SentenceTransformer


# Dataset oku
df = pd.read_csv("../data/processed_dataset.csv")

# Embedding modeli yükle
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Clean textleri al
texts = df["clean_text"].tolist()

# Embedding üret
embeddings = model.encode(texts)

# İlk embedding
print("İlk embedding:\n")
print(embeddings[0][:10])

# Kaç embedding oluştu
print("\nToplam embedding sayısı:")
print(len(embeddings))

# Bir embedding boyutu
print("\nEmbedding boyutu:")
print(len(embeddings[0]))