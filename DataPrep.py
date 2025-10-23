import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
