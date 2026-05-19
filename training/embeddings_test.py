from sentence_transformers import SentenceTransformer


# Model yükle
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Test cümlesi
text = "Harika yine internet gitti"

# Embedding oluştur
embedding = model.encode(text)

# İlk birkaç sayı
print(embedding[:10])

# Vektör uzunluğu
print("\nVector length:")
print(len(embedding))