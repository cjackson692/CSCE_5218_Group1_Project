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
from TranslationModel import *

with open('en_padded.pickle', 'rb') as f:
    en_padded = pickle.load(f)

with open('tl_padded.pickle', 'rb') as f:
    tl_padded = pickle.load(f)

with open('en_tokenizer.pickle', 'rb') as f:
    en_tokenizer = pickle.load(f)

with open('tl_tokenizer.pickle', 'rb') as f:
    tl_tokenizer = pickle.load(f)


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

#Need to tune hyperparameters
num_layers = 4
num_heads = 4
num_kv_heads = 4
hidden_dim = 64
max_seq_len = max(max_en_len, max_tl_len)
vocab_size_in = len(en_tokenizer.word_index)+1
vocab_size_out = len(tl_tokenizer.word_index)+1
dropout = .25

model = Transformer(num_layers, num_heads, num_kv_heads, hidden_dim, max_seq_len, vocab_size_in, vocab_size_out, dropout) # this is assuming we use the transformer class in the example_model.py script

loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

n_epochs = 60
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



import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losscurves')

truths = []
preds = []
for element in range(len(test_dataset)):
  with torch.no_grad():
      start_token = torch.tensor([1]).to(device)
      en_ids, true_tl = test_dataset[element]

      en_ids = en_ids.unsqueeze(0).to(device)
      src_mask = create_padding_mask(en_ids, pad_token_id)
      x = model.src_embedding(en_ids)
      for encoder in model.encoders:
          x = encoder(x, src_mask, model.rope)
      enc_out = x
      tl_ids = start_token.unsqueeze(0).to(device)
      for _ in range(max_len):
        tgt_mask = create_causal_mask(tl_ids.shape[0], device).unsqueeze(0) + create_padding_mask(tl_ids, pad_token_id)
        x = model.tgt_embedding(tl_ids)
        for decoder in model.decoders:
          x = decoder(x, enc_out, tgt_mask, model.rope)
        outputs = model.out(x)
        outputs = outputs.argmax(dim=-1)
        tl_ids = torch.cat([tl_ids, outputs[:, -1:]], axis=-1)
        if tl_ids[0, -1] == 2:
            break

          # Decode the predicted IDs
  pred_tl = tl_tokenizer.sequences_to_texts([tl_ids[0].tolist()])

  truths.append(tl_tokenizer.sequences_to_texts([true_tl.tolist()]))
  preds.append(pred_tl)


with open('truths.pickle', 'wb') as pkl_file:
    pickle.dump(truths, pkl_file)

with open('preds.pickle', 'wb') as pkl_file:
    pickle.dump(preds, pkl_file)

torch.save(train_dataloader, 'train_dataloader.pt')
torch.save(test_dataloader, 'test_dataloader.pt')

model.eval()
max_len = max_seq_len
samples = random.sample(range(len(test_dataset)), 5)
for element in samples:
  with torch.no_grad():
      start_token = torch.tensor([1]).to(device)
      en_ids, true_tl = test_dataset[element]

      en_ids = en_ids.unsqueeze(0).to(device)
      src_mask = create_padding_mask(en_ids, pad_token_id)
      x = model.src_embedding(en_ids)
      for encoder in model.encoders:
          x = encoder(x, src_mask, model.rope)
      enc_out = x
      tl_ids = start_token.unsqueeze(0).to(device)
      for _ in range(max_len):
        tgt_mask = create_causal_mask(tl_ids.shape[0], device).unsqueeze(0) + create_padding_mask(tl_ids, pad_token_id)
        x = model.tgt_embedding(tl_ids)
        for decoder in model.decoders:
          x = decoder(x, enc_out, tgt_mask, model.rope)
        outputs = model.out(x)
        outputs = outputs.argmax(dim=-1)
        tl_ids = torch.cat([tl_ids, outputs[:, -1:]], axis=-1)
        if tl_ids[0, -1] == 2:
            break

          # Decode the predicted IDs
  pred_tl = tl_tokenizer.sequences_to_texts([tl_ids[0].tolist()])
  print(f"English: {en_tokenizer.sequences_to_texts([en_ids[0].tolist()])}")
  print(f"True Tagalog: {tl_tokenizer.sequences_to_texts([true_tl.tolist()])}")
  print(f"Predicted: {pred_tl}")
  print()
