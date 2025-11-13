import pickle
import sacrebleu

with open('preds.pickle', 'rb') as f:
    raw_preds = pickle.load(f)


with open('truths.pickle', 'rb') as f:
    raw_truths = pickle.load(f)

def strip_sentence_markers(preds, truths):
    predictions = [p[0] if isinstance(p, list) else p for p in preds]
    references = [g[0] if isinstance(g, list) else g for g in truths]

    predictions = [p.replace("<sos>", "").replace("<eos>", "").strip() for p in predictions]
    references = [g.replace("<sos>", "").replace("<eos>", "").strip() for g in references]
    return predictions, [references]

predictions, references = strip_sentence_markers(raw_preds, raw_truths)
# Now BLEU works
bleu = sacrebleu.corpus_bleu(predictions, references)
print(bleu)
