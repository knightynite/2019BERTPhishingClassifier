"""Evaluation script for the phishing classifier.

Computes classification report (precision/recall/F1 per class), confusion
matrix, and ROC curve. Writes a JSON summary + PNG plots to results/.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from transformers import BertTokenizer, BertForSequenceClassification

from data import EmailDataset


def collect_predictions(model, tokenizer, csv_path, device, batch_size=32, max_len=256):
    ds = EmailDataset(csv_path, tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size)
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            out = model(input_ids, attention_mask=attention_mask)
            logits = out[0] if isinstance(out, tuple) else out.logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)


def maybe_plot(probs, labels, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.set_title('Phishing classifier — ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'roc.png'))
    plt.close(fig)

    cm = confusion_matrix(labels, probs.argmax(axis=1))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['benign', 'phishing'])
    ax.set_yticklabels(['benign', 'phishing'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Phishing classifier — Confusion matrix')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'confusion_matrix.png'))
    plt.close(fig)
    return roc_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--test-csv', required=True)
    p.add_argument('--out', default='results/eval_summary.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)

    probs, labels = collect_predictions(model, tokenizer, args.test_csv, device)

    preds = probs.argmax(axis=1)
    report = classification_report(labels, preds,
                                   target_names=['benign', 'phishing'],
                                   output_dict=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    auc_val = maybe_plot(probs, labels, os.path.dirname(args.out))

    summary = {
        'classification_report': report,
        'auc': auc_val,
        'n_examples': int(len(labels)),
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
