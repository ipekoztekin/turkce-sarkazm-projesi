import pandas as pd

from preprocess import clean_text


# CSV dosyalarını oku
normal_df = pd.read_csv("../data/manual/normal.csv")
irony_df = pd.read_csv("../data/manual/irony.csv")
mixed_df = pd.read_csv("../data/manual/mixed.csv")

# Birleştir
df = pd.concat([
    normal_df,
    irony_df,
    mixed_df
])

# Karıştır
df = df.sample(frac=1).reset_index(drop=True)

# Temizleme uygula
df["clean_text"] = df["text"].apply(clean_text)

# İlk satırlar
print(df.head())

# Yeni CSV kaydet
df.to_csv(
    "../data/processed_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("\nDataset kaydedildi.")