import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, unicodedata

# Pull Data
with open("OpenSubtitles.en-tl.en", encoding="utf-8") as f_en:
    en_sentences = [line.strip() for line in f_en]

with open("OpenSubtitles.en-tl.tl", encoding="utf-8") as f_tl:
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

# After cleaning, remove rows that are empty
df = df[(df["english"] != "") & (df["tagalog"] != "")]
len2 = len(df)
print(f"Rows kept after cleaning: {len2}/{len1}")

# Add SOS and EOS
df['tagalog'] = df['tagalog'].apply(lambda x: f"<SOS> {x} <EOS>")

# Tokenize each df
en_tokenizer = Tokenizer(filters='', lower=True)
tl_tokenizer = Tokenizer(filters='', lower=True)

# Converts Word to integer mapping
en_tokenizer.fit_on_texts(df['english'])
tl_tokenizer.fit_on_texts(df['tagalog'])

# Convert Sentence to integers
en_sequences = en_tokenizer.texts_to_sequences(df['english'])
tl_sequences = tl_tokenizer.texts_to_sequences(df['tagalog'])

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
