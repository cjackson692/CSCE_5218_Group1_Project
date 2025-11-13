import comet
from comet import download_model, load_from_checkpoint
import torch
import pickle

model_path = download_model("Unbabel/wmt22-comet-da")
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = load_from_checkpoint(model_path)
model = model.to(device)

with open('preds.pickle', 'rb') as f:
    preds = pickle.load(f)

with open('sources.pickle', 'rb') as f:
    source = pickle.load(f)

with open('truths.pickle', 'rb') as f:
    truth = pickle.load(f)

def strip_sentence_markers(preds, truths, source):
    predictions = [p[0] if isinstance(p, list) else p for p in preds]
    references = [g[0] if isinstance(g, list) else g for g in truths]
    sources = [r[0] if isinstance(r, list) else r for r in source]

    predictions = [p.replace("<sos>", "").replace("<eos>", "").strip() for p in predictions]
    references = [g.replace("<sos>", "").replace("<eos>", "").strip() for g in references]
    sources = [r.replace("<sos>", "").replace("<eos>", "").strip() for r in sources]
    return predictions, references, sources

# Apply it now
predictions, truths, sources = strip_sentence_markers(preds, truth, source)

data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, truths)]

scores = model.predict(data, batch_size=8, gpus=1 if device == "cuda" else 0)

# Calculate average COMET score
avg_score = sum(scores[0]) / len(scores[0])
print(f"Average COMET score: {avg_score:.4f}")

# Optional: Print per-sentence scores
for i, score in enumerate(scores[0]):
    print(f"Sentence {i+1}: {score:.4f}")
