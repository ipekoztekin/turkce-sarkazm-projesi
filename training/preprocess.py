import re


def clean_text(text):

    # Küçük harf
    text = text.lower()

    # Link sil
    text = re.sub(r"http\S+|www\S+", "", text)

    # Mention sil
    text = re.sub(r"@\w+", "", text)

    # Korunacak yapılar
    special_tokens = [
        ":ddd",
        ":dd",
        ":d",
        ":c",
        "??",
        "(?)",
        "!!",
        "(!)",
        "..."
    ]

    token_map = {}

    # Regex-safe placeholder
    for i, token in enumerate(special_tokens):

        placeholder = f"tokennumber{i}"

        # re.escape -> regex karakterlerini güvenli yapar
        text = re.sub(
            re.escape(token),
            placeholder,
            text
        )

        token_map[placeholder] = token

    # Harf tekrarlarını max 3 yap
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

    # Gereksiz karakterleri temizle
    text = re.sub(
        r"[^a-zçğıöşü0-9\s]",
        "",
        text
    )

    # Fazla boşluk temizle
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenleri geri koy
    for placeholder, token in token_map.items():

        text = text.replace(
            placeholder,
            token
        )

    return text


# TEST

test_text = "YHAAAAAAA :DDDD ???? harikasın..... (?)"

print(clean_text(test_text))