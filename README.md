# 2019 — BERT Phishing Email Classifier

Fine-tune BERT-base-uncased for binary classification of phishing vs. benign emails.

## Approach

1. Tokenize emails with BERT's WordPiece tokenizer (`bert-base-uncased`)
2. Add a single linear classification head on top of `[CLS]`
3. Fine-tune end-to-end on labeled phishing dataset
4. Evaluate accuracy, precision/recall, F1

## Files

- `src/bert_classifier.py` — BertForSequenceClassification fine-tune loop
- `src/data.py` — dataset loader (placeholder — drop your CSV at `data/emails.csv`)

## Run

```bash
pip install -r requirements.txt
python src/bert_classifier.py --train data/emails.csv --epochs 3
```

## Dataset

Use Nazario Phishing Corpus (public, classic) or any labeled email CSV with columns
`text` and `label` (0 = benign, 1 = phishing). Don't commit real emails — this is for
local study only.

