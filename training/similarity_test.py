import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Dataset oku
df = pd.read_csv("../data/processed_dataset.csv")

# Model yükle
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Dataset textleri
texts = df["clean_text"].tolist()

# Embedding üret
embeddings = model.encode(texts)

# Kullanıcı cümlesi
query = "harika yine sular kesildi"

# Query embedding
query_embedding = model.encode([query])

# Similarity hesapla
scores = cosine_similarity(
    query_embedding,
    embeddings
)

# En benzer index
best_match_index = scores.argmax()

# Sonuç
print("Kullanıcı cümlesi:\n")
print(query)

print("\nEn benzer dataset cümlesi:\n")
print(df.iloc[best_match_index]["text"])

print("\nSimilarity score:")
print(scores[0][best_match_index])