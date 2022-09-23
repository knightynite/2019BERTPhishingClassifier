# Architecture

## Module map

```
src/
├── data.py          EmailDataset + stratified_split
├── utils.py         email normalization (RFC 822 → body → tokens)
├── bert_classifier.py   training loop (AdamW + linear warmup)
├── predict.py       CLI inference (text or CSV)
└── evaluate.py      classification report + ROC + CM
```

## Pipeline

```
raw email (mbox/csv)
     │
     ▼
utils.normalize_email   → strip headers, decode QP, replace URL/IP/email
     │
     ▼
data.EmailDataset       → tokenize with bert-base-uncased
     │
     ▼
bert_classifier.train   → fine-tune (AdamW + warmup, 3 epochs default)
     │
     ▼
checkpoints/
     │
     ├──→ predict.py   (CLI inference)
     └──→ evaluate.py  (held-out metrics + plots)
```

## Why neutralize URLs / IPs / emails

`utils.neutralize` replaces concrete URL/IP/email strings with `<URL>`,
`<IP>`, `<EMAIL>` placeholders before tokenization. Two competing pulls:

- Concrete strings carry signal — attacker reuses infrastructure.
- Concrete strings cause overfit — the model memorizes specific
  campaigns and degrades on novel ones.

Neutralization is the conservative choice. Ablate by toggling the flags
in `utils.neutralize` to compare.
