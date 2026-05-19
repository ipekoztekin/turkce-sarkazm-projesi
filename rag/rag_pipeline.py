import chromadb
import ollama

from sentence_transformers import SentenceTransformer


# Embedding modeli
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ChromaDB bağlantısı
client = chromadb.PersistentClient(
    path="../vector_db"
)

collection = client.get_collection(
    name="sarcasm_dataset"
)

# Kullanıcı sorgusu
query = "harika yine internet kesildi"

# Query embedding
query_embedding = model.encode(query).tolist()

# Benzer örnekleri getir
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_docs = results["documents"][0]

# Context oluştur
context = "\n".join(retrieved_docs)

# Prompt
prompt = f"""
Sen Türkçe sarkazm analizi yapan bir AI sistemisin.

Kullanıcı cümlesi:
{query}

Benzer örnekler:
{context}

Bu cümle sarkastik mi?
Kısa açıklama yap.
"""

# Qwen çağır
response = ollama.chat(
    model="qwen2.5",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Sonuç
print("\nKULLANICI SORGUSU:\n")
print(query)

print("\nRETRIEVED CONTEXT:\n")

for doc in retrieved_docs:
    print("-", doc)

print("\nQWEN YORUMU:\n")
print(response["message"]["content"])