import chromadb

from sentence_transformers import SentenceTransformer


# Embedding modeli
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Chroma client
client = chromadb.PersistentClient(
    path="../vector_db"
)

# Collection al
collection = client.get_collection(
    name="sarcasm_dataset"
)

# Kullanıcı sorgusu
query = "harika yine internet kesildi"

# Query embedding
query_embedding = model.encode(query).tolist()

# Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nKullanıcı sorgusu:\n")
print(query)

print("\nEn benzer sonuçlar:\n")

for i, doc in enumerate(results["documents"][0]):

    print(f"{i+1}. {doc}")