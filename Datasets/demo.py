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


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = x
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.cross_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)

    def forward(self, x, enc_out, mask=None, rope=None):
        # self-attention sublayer
        out = x
        out = self.norm1(out)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # cross-attention sublayer
        out = self.norm2(x)
        out = self.cross_attn(out, enc_out, enc_out, None, rope)
        x = out + x
        # MLP sublayer
        x = out + x
        out = self.norm3(x)
        out = self.mlp(out)
        return out + x

## The actual model we ahould be able to use
class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size_src, vocab_size_tgt, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.src_embedding = nn.Embedding(vocab_size_src, hidden_dim)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, hidden_dim)
        self.encoders = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, vocab_size_tgt)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Encoder
        x = self.src_embedding(src_ids)
        for encoder in self.encoders:
            x = encoder(x, src_mask, self.rope)
        enc_out = x
        # Decoder
        x = self.tgt_embedding(tgt_ids)
        for decoder in self.decoders:
            x = decoder(x, enc_out, tgt_mask, self.rope)
        return self.out(x)


def load_model(model_path, device, en_tokenizer, tl_tokenizer):
    print("Loading model...")

    num_layers = 4
    num_heads = 8
    num_kv_heads = 4
    hidden_dim = 128
    max_seq_len = 75
    vocab_size_in = len(en_tokenizer.word_index)+1
    vocab_size_out = len(tl_tokenizer.word_index)+1
    dropout = .1

    model = Transformer(num_layers, num_heads, num_kv_heads, hidden_dim, max_seq_len, vocab_size_in, vocab_size_out, dropout) # this is assuming we use the transformer class in the example_model.py script
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 3. Move to device and eval mode
    model.to(device)
    model.eval()

    print("Model loaded successfully.\n")
    return model

# ------------------------------------------------------------------
#  Translation Function
# ------------------------------------------------------------------
def translate_sentence(model, sentence, en_tokenizer, tl_tokenizer, device, max_len, pad_token_id):
    model.eval()

    # Tokenize input
    en_ids = en_tokenizer.texts_to_sequences([sentence])[0]
    en_ids = torch.tensor(en_ids).unsqueeze(0).to(device)

    # Encoder
    src_mask = create_padding_mask(en_ids, pad_token_id)
    x = model.src_embedding(en_ids)
    for encoder in model.encoders:
        x = encoder(x, src_mask, model.rope)
    enc_out = x

    # Decoder initialization
    start_token = torch.tensor([[1]]).to(device)  # <start>
    tl_ids = start_token.clone()

    # Autoregressive decoding
    for _ in range(max_len):
        tgt_mask = (
            create_causal_mask(tl_ids.shape[-1], device).unsqueeze(0)
            + create_padding_mask(tl_ids, pad_token_id)
        )
        x = model.tgt_embedding(tl_ids)
        for decoder in model.decoders:
            x = decoder(x, enc_out, tgt_mask, model.rope)

        outputs = model.out(x)
        next_token = outputs.argmax(dim=-1)[:, -1:]
        tl_ids = torch.cat([tl_ids, next_token], dim=-1)

        if next_token.item() == 2:  # <end> token
            break

    # Decode tokens
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_out.pth"

    with open('en_tokenizer.pickle', 'rb') as f:
        en_tokenizer = pickle.load(f)

    with open('tl_tokenizer.pickle', 'rb') as f:
        tl_tokenizer = pickle.load(f)

    # Load your model
    model = load_model(model_path, device, en_tokenizer, tl_tokenizer)


    pad_toke_id = 0
    max_seq_len = 75

    print("Interactive English → Tagalog translation.\nType 'quit' to exit.\n")

    while True:
        sentence = input("Enter an English sentence: ").strip()
        if sentence.lower() in {"quit", "exit"}:
            print("Exiting translator.")
            break
        if not sentence:
            continue

        try:
            translation = translate_sentence(
                model, sentence, en_tokenizer, tl_tokenizer, device, max_seq_len, pad_token_id
            )
            print(f"🗣️ English: {sentence}")
            print(f"🇵🇭 Tagalog Translation: {translation}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()
