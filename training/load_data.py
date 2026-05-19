import pandas as pd

# CSV dosyalarını oku
normal_df = pd.read_csv("../data/manual/normal.csv")
irony_df = pd.read_csv("../data/manual/irony.csv")
mixed_df = pd.read_csv("../data/manual/mixed.csv")

# Hepsini birleştir
df = pd.concat([normal_df, irony_df, mixed_df])

# Karıştır
df = df.sample(frac=1).reset_index(drop=True)

# İlk 5 satır
print(df.head())

# Veri sayısı
print("\nToplam veri sayısı:")
print(len(df))

# Label dağılımı
print("\nLabel dağılımı:")
print(df["label"].value_counts())