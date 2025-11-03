import pandas as pd
import re
import unicodedata
import pickle
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import PreTrainedTokenizerFast
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import random

# Pull Data
with open("OpenSubtitles.en-tl.en", encoding="utf-8") as f_en:
    en_sentences = [line.strip() for line in f_en]

with open("OpenSubtitles.en-tl.tl", encoding="utf-8") as f_tl:
    tl_sentences = [line.strip() for line in f_tl]

# Ensures same number of lines
num_seq = 200000
en_sentences = en_sentences[:num_seq]
tl_sentences = tl_sentences[:num_seq]


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



# Define Special Tokens
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "<SOS>" # For Tagalog (Decoder Input)
EOS_TOKEN = "<EOS>" # For both

special_tokens = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
vocab_size = 30000 

def train_bpe_tokenizer(data, save_path, special_tokens, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() 
    trainer = trainers.BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        min_frequency=2 
    )
    
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.save(save_path)
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=SOS_TOKEN,
        eos_token=EOS_TOKEN,
    )
    
    return wrapped_tokenizer


en_tokenizer = train_bpe_tokenizer(df['english'].tolist(), 'en_bpe_tokenizer.json', special_tokens, vocab_size)
tl_tokenizer = train_bpe_tokenizer(df['tagalog'].tolist(), 'tl_bpe_tokenizer.json', special_tokens, vocab_size)

all_en_tokens = en_tokenizer(df['english'].tolist(), add_special_tokens=False)['input_ids']
all_tl_tokens = tl_tokenizer(df['tagalog'].tolist(), add_special_tokens=True)['input_ids']

max_en_len = max(len(seq) for seq in all_en_tokens) if all_en_tokens else 0
max_tl_len = max(len(seq) for seq in all_tl_tokens) if all_tl_tokens else 0

print(f"Max English sentence length (tokens): {max_en_len}")
print(f"Max Tagalog sentence length (tokens): {max_tl_len}")

tl_encoding = tl_tokenizer(
    df['tagalog'].tolist(), 
    max_length=max_tl_len, 
    padding='max_length', 
    truncation=True,
    return_tensors='np',
    add_special_tokens=True 
)
tl_padded = tl_encoding['input_ids']

en_encoding = en_tokenizer(
    df['english'].tolist(), 
    max_length=max_en_len, 
    padding='max_length', 
    truncation=True,
    return_tensors='np', 
    add_special_tokens=False 
)
en_padded = en_encoding['input_ids']

print("English padded shape:", en_padded.shape)
print("Tagalog padded shape:", tl_padded.shape)

with open('en_padded_bpe.pickle', 'wb') as handle:
    pickle.dump(en_padded, handle)
with open('tl_padded_bpe.pickle', 'wb') as handle:
    pickle.dump(tl_padded, handle)
