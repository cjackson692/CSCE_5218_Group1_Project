import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, unicodedata
import zipfile
import os
import pickle

zip_file_path = 'Datasets/Cleaned Data.zip'
extraction_directory = 'Datasets/Clean'
os.makedirs(extraction_directory, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_directory)

# Pull Data
with open("Datasets/Clean/Cleaned Data/clean.en-tl.en", encoding="utf-8") as f_en:
    en_sentences = [line.strip() for line in f_en]

with open("Datasets/Clean/Cleaned Data/clean.en-tl.tl", encoding="utf-8") as f_tl:
    tl_sentences = [line.strip() for line in f_tl]

# Ensures same number of lines
min_len = min(len(en_sentences), len(tl_sentences))
en_sentences = en_sentences[:min_len]
tl_sentences = tl_sentences[:min_len]

# Make it to where we only accept files w/ same number of lines later

# Merge data frames
df = pd.DataFrame({
    'english': en_sentences,
    'tagalog': tl_sentences
})

# Drop null lines
df = df[(df['english'].str.strip() != '') & (df['tagalog'].str.strip() != '')].dropna()

def clean(data):
    # Normalize unicode and punctutation
    data = unicodedata.normalize("NFKC", data)
    data = data.replace("—", "-").replace("–", "-").replace("？", "?")

    # Remove unnecessary special characters and normalize whitespace
    data = re.sub(r"[\x00-\x1F\x80-\x9F]", " ", data)
    data = re.sub(r"[\u200b\u200e\u202a]", " ", data)
    data = re.sub(r"[\u0370-\u03FF]", " ", data)
    data = re.sub(r"[\u0900-\u097F]", " ", data)
    data = re.sub(r"[\u0D80-\u0DFF]", " ", data)
    data = re.sub(r"[\u0400-\u04FF]", " ", data)
    data = re.sub(r"[\u4E00-\u9FFF]", " ", data)
    data = re.sub(r"[¢£¤¥¦§©ª®¯°±²³¶¸º¼½¾‰♡♥♪♬✰€]+", " ", data)
    data = re.sub(r"<[^>]+>", " ", data)
    data = re.sub(r"\s+", " ", data).strip()

    return data

# Clean data
len1 = len(df)
df["english"] = df["english"].apply(clean)
df["tagalog"] = df["tagalog"].apply(clean)

def strip_punctuation_regex(series):
    return (
        series.astype(str)
              .str.replace(r"[^\w\s]", " ", regex=True)  
              .str.lower()
              .str.replace(r"\s+", " ", regex=True)      
              .str.strip()
    )

df["english"] = strip_punctuation_regex(df["english"])
df["tagalog"] = strip_punctuation_regex(df["tagalog"])

# After cleaning, remove rows that are empty
df = df[(df["english"] != "") & (df["tagalog"] != "")]
len2 = len(df)
print(f"Rows kept after cleaning: {len2}/{len1}")

# Ensure empty and whitespace strings are removed
df = df[(df["english"].str.len() > 0) & (df["tagalog"].str.len() > 0)]
empty = df[(df["english"] == "") | (df["tagalog"] == "")]
print(f"Empty or whitespace string lines: {len(empty)}\n")

# Add SOS and EOS
df['tagalog'] = df['tagalog'].apply(lambda x: f"<sos> {x} <eos>")


temp_A = []
temp_B = []
for seq_A, seq_B in zip(list(df["english"]), list(df["tagalog"])):
    if len(seq_A) <= 50 and len(seq_B) <= 50:
        temp_A.append(seq_A)
        temp_B.append(seq_B)
en_sentences = temp_A[:500000]
tl_sentences= temp_B[:500000]

# Tokenize each df
en_tokenizer = Tokenizer(filters='', lower=True)
tl_tokenizer = Tokenizer(filters='', lower=True)

# Converts Word to integer mapping
en_tokenizer.fit_on_texts(en_sentences)
tl_tokenizer.fit_on_texts(tl_sentences)

# Convert Sentence to integers
en_sequences = en_tokenizer.texts_to_sequences(en_sentences)
tl_sequences = tl_tokenizer.texts_to_sequences(tl_sentences)



# Find Max Lengths (used for padding)
max_en_len = max(len(seq) for seq in en_sequences)
max_tl_len = max(len(seq) for seq in tl_sequences)

# Pad Sequence
en_padded = pad_sequences(en_sequences, maxlen=max_en_len, padding='post')
tl_padded = pad_sequences(tl_sequences, maxlen=max_tl_len, padding='post')

print("English padded shape:", en_padded.shape)
print("Tagalog padded shape:", tl_padded.shape)

with open('en_tokenizer.pickle', 'wb') as handle:
    pickle.dump(en_tokenizer, handle)
with open('tl_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tl_tokenizer, handle)
with open('en_padded.pickle', 'wb') as handle:
    pickle.dump(en_padded, handle)
with open('tl_padded.pickle', 'wb') as handle:
    pickle.dump(tl_padded, handle)
