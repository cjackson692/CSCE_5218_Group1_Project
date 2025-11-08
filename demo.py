from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import random
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
from TranslationModel import *
import os
import time

# Set a new environment variable or modify an existing one
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_model(model_path, device, en_tokenizer, tl_tokenizer):
    print("Loading model...")

    num_layers = 6
    num_heads = 16
    num_kv_heads = 16
    hidden_dim = 256
    max_seq_len = 1024
    vocab_size_in = len(en_tokenizer.word_index)+1
    vocab_size_out = len(tl_tokenizer.word_index)+1
    dropout = .5

    model = Transformer(num_layers, num_heads, num_kv_heads, hidden_dim, max_seq_len, vocab_size_in, vocab_size_out, dropout) 
    model.load_state_dict(torch.load(model_path, map_location=device))


    model.to(device)
    model.eval()

    print("Model loaded successfully.\n")
    return model

def translate_sentence(model, sentence, en_tokenizer, tl_tokenizer, device, max_len, pad_token_id):
    model.eval()
    with torch.no_grad():
        en_ids = pad_sequences([en_tokenizer.texts_to_sequences([sentence])[0]], maxlen=max_len, padding='post')[0]
        #print(en_ids)
        en_ids = torch.tensor(en_ids).unsqueeze(0).to(device)

        src_mask = create_padding_mask(en_ids, pad_token_id)
        x = model.src_embedding(en_ids)
        for encoder in model.encoders:
            x = encoder(x, src_mask, model.rope)
        enc_out = x
    
        start_token = torch.tensor([[1]]).to(device)
        tl_ids = start_token.clone()

        for _ in range(max_len):
            tgt_mask = (
                create_causal_mask(tl_ids.shape[0], device).unsqueeze(0)
                + create_padding_mask(tl_ids, pad_token_id)
            )
            x = model.tgt_embedding(tl_ids)
            for decoder in model.decoders:
                x = decoder(x, enc_out, tgt_mask, model.rope)

            outputs = model.out(x)
            next_token = outputs.argmax(dim=-1)
            tl_ids = torch.cat([tl_ids, next_token[:, -1:]], axis=-1)
            #print(tl_tokenizer.sequences_to_texts([next_token[:, -1:][0].tolist()])[0], end = ' ')
            #time.sleep(1)
            if tl_ids[0, -1] == 2:
                break

    pred_tl = tl_tokenizer.sequences_to_texts([tl_ids[0].tolist()])[0]
    return pred_tl

def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

def create_padding_mask(batch, padding_token_id):
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float().masked_fill(batch == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]

def strip_punctuation_regex(string_list):
    cleaned_strings = [re.sub(r'[^\w\s]', '', s).lower() for s in list(string_list)]
    return cleaned_strings



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_out.pth"

    with open('en_tokenizer.pickle', 'rb') as f:
        en_tokenizer = pickle.load(f)

    with open('tl_tokenizer.pickle', 'rb') as f:
        tl_tokenizer = pickle.load(f)

    model = load_model(model_path, device, en_tokenizer, tl_tokenizer)


    pad_token_id = 0
    max_seq_len = 25

    print("Interactive English → Tagalog translation.\nType 'quit' to exit.\n")

    while True:
            sentence = input("Enter an English sentence: ").strip().lower()
            sentence = strip_punctuation_regex([sentence])[0]

            #print(sentence)
            if sentence.lower() in {"quit", "exit"}:
                print("Exiting translator.")
                break
            if not sentence:
                continue
            #sentence = '<sos> ' + sentence + ' <eos>'
            try:
                translation = translate_sentence(
                    model, sentence, en_tokenizer, tl_tokenizer, device, max_seq_len, pad_token_id
                )
                print()
                print(f"🗣️ English: {sentence}")
                print(f"🇵🇭 Tagalog Translation: {translation}\n")
            except Exception as e:
                print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()
