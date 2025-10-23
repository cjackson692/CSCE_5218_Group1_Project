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


with open('en_padded.pickle', 'rb') as f:
    en_padded = pickle.load(f)

with open('tl_padded.pickle', 'rb') as f:
    tl_padded = pickle.load(f)

with open('en_tokenizer.pickle', 'rb') as f:
    en_tokenizer = pickle.load(f)

with open('tl_tokenizer.pickle', 'rb') as f:
    tl_tokenizer = pickle.load(f)


en_padded = np.load('en_padded.npy')
tl_padded = np.load('tl_padded.npy')
max_en_len = en_padded.shape[1]
max_tl_len = tl_padded.shape[1]

source_train, source_test, target_train, target_test = train_test_split(
    en_padded,
    tl_padded,
    test_size=0.1,
    random_state=42 
)

print(f"Source Train shape: {source_train.shape}")
print(f"Target Train shape: {target_train.shape}")
print(f"Source Test shape: {source_test.shape}")
print(f"Target Test shape: {target_test.shape}")

source_train_tensor = torch.from_numpy(source_train).long()
target_train_tensor = torch.from_numpy(target_train).long()
source_test_tensor = torch.from_numpy(source_test).long()
target_test_tensor = torch.from_numpy(target_test).long()

train_dataset = TensorDataset(source_train_tensor, target_train_tensor)
test_dataset = TensorDataset(source_test_tensor, target_test_tensor)

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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



#Need to tune hyperparameters
num_layers = 4
num_heads = 8
num_kv_heads = 4
hidden_dim = 128
max_seq_len = max(max_en_len, max_tl_len)
vocab_size_in = len(en_tokenizer.word_index)+1
vocab_size_out = len(tl_tokenizer.word_index)+1
dropout = .2

model = Transformer(num_layers, num_heads, num_kv_heads, hidden_dim, max_seq_len, vocab_size_in, vocab_size_out, dropout) # this is assuming we use the transformer class in the example_model.py script

loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

n_epochs = 30
lr = 5e-4
n_warmup = 1000
gradient_clip = 5.0
best_loss = float('inf')
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * len(train_dataloader) - n_warmup)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[n_warmup])

def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

def create_padding_mask(batch, padding_token_id):
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float().masked_fill(batch == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]

pad_token_id = 0

train_losses = []
test_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for source_lang, target_lang in tqdm.tqdm(train_dataloader):
        source_lang = source_lang.to(device)
        target_lang = target_lang.to(device)
        src_mask = create_padding_mask(source_lang, pad_token_id)
        tgt_mask = create_causal_mask(target_lang.shape[1], device).unsqueeze(0) + create_padding_mask(target_lang, pad_token_id)
        optimizer.zero_grad()
        outputs = model(source_lang, target_lang, src_mask, tgt_mask)
        loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), target_lang[:, 1:].reshape(-1))
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=False)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}; Avg loss {epoch_loss/len(train_dataloader)}; Latest loss {loss.item()}")
    train_losses.append(epoch_loss/len(train_dataloader))
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for source_lang, target_lang in tqdm.tqdm(test_dataloader):
            source_lang = source_lang.to(device)
            target_lang = target_lang.to(device)
            src_mask = create_padding_mask(source_lang, pad_token_id)
            tgt_mask = create_causal_mask(target_lang.shape[1], device).unsqueeze(0) + create_padding_mask(target_lang, pad_token_id)
            outputs = model(source_lang, target_lang, src_mask, tgt_mask)
            loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), target_lang[:, 1:].reshape(-1))
            epoch_loss += loss.item()
    print(f"Eval loss: {epoch_loss/len(test_dataloader)}")
    test_losses.append(epoch_loss/len(test_dataloader))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "model_out.pth")