"""CLI for running inference with a fine-tuned phishing classifier.

Usage:
    python src/predict.py --model-dir checkpoints/bert-phishing/ --text "..."
    python src/predict.py --model-dir checkpoints/bert-phishing/ --csv emails.csv
"""
import argparse
import csv
import sys

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_model(model_dir, device):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def predict_one(tokenizer, model, text, device, max_len=256):
    enc = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)
    logits = out[0] if isinstance(out, tuple) else out.logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = int(torch.argmax(probs).item())
    return pred, probs.cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--text', help='single email text to classify')
    p.add_argument('--csv', help='CSV with `text` column')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='probability threshold for phishing class')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = load_model(args.model_dir, device)

    if args.text:
        pred, probs = predict_one(tokenizer, model, args.text, device)
        label = 'phishing' if probs[1] >= args.threshold else 'benign'
        print(f"{label}\tphishing_prob={probs[1]:.4f}")
        return

    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(['idx', 'pred', 'phishing_prob'])
        with open(args.csv) as f:
            for i, row in enumerate(csv.DictReader(f)):
                pred, probs = predict_one(tokenizer, model, row['text'], device)
                label = 'phishing' if probs[1] >= args.threshold else 'benign'
                w.writerow([i, label, f"{probs[1]:.4f}"])
        return

    print('error: provide --text or --csv', file=sys.stderr)
    sys.exit(2)


if __name__ == '__main__':
    main()
