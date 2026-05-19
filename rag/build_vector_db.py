import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


# Dataset oku
df = pd.read_csv("../data/processed_dataset.csv")

# Embedding modeli
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Chroma client
client = chromadb.PersistentClient(
    path="../vector_db"
)

# Collection oluştur
collection = client.get_or_create_collection(
    name="sarcasm_dataset"
)

# Textler
texts = df["clean_text"].tolist()

# Embedding üret
embeddings = model.encode(texts).tolist()

# ChromaDB'ye ekle
for i, row in df.iterrows():

    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i]],
        documents=[row["text"]],
        metadatas=[{
            "label": int(row["label"])
        }]
    )

print("Vector DB oluşturuldu.")