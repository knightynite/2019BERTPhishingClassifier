# 2019 — BERT Phishing Email Classifier

Fine-tune BERT-base-uncased for binary classification of phishing vs. benign emails.

## Approach

1. Tokenize emails with BERT's WordPiece tokenizer (`bert-base-uncased`)
2. Add a single linear classification head on top of `[CLS]`
3. Fine-tune end-to-end on labeled phishing dataset
4. Evaluate accuracy, precision/recall, F1
