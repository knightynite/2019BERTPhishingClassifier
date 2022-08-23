"""BERT fine-tune for phishing classification. transformers 2.x style."""
import argparse

import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report

from data import EmailDataset


def train(csv_path, epochs=3, batch_size=16, lr=2e-5, max_len=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    ).to(device)

    ds = EmailDataset(csv_path, tokenizer, max_len=max_len)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    total_steps = len(train_loader) * epochs
    optim = AdamW(model.parameters(), lr=lr, correct_bias=False)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=0,
                                            num_training_steps=total_steps)

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(ids, attention_mask=mask, labels=labels)
            loss = outputs[0]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, attention_mask=mask)[0]
                ps += logits.argmax(-1).cpu().tolist()
                ys += batch['label'].tolist()
        print('=== epoch {} ==='.format(ep + 1))
        print(classification_report(ys, ps, target_names=['benign', 'phishing']))

    model.save_pretrained('./bert_phishing')
    tokenizer.save_pretrained('./bert_phishing')
    print('Saved to ./bert_phishing')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, help='Path to training CSV')
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    train(args.train, epochs=args.epochs)
