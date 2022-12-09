# Dataset notes

The model in this repo expects CSVs with two columns: `text` and `label`
(0 = benign, 1 = phishing). What you put in those CSVs matters more than
the model architecture. A few notes from training/evaluation cycles.

## Suggested public corpora

- **Nazario Phishing Corpus.** Decade-long collection of reported phishing
  emails. Format: mbox; convert to CSV (one email per row, body in `text`).
  Stable phishing baseline.
- **Enron email dataset.** The benign half. Sample with rough class balance —
  Enron alone is ~500K messages, so under-sample for a manageable training
  set.
- **SpamAssassin public corpus.** Older but well-labeled "ham" + "spam" split.
  Spam ≠ phishing exactly, but it's a reasonable proxy for "unwanted
  commercial mail" and pulls the model toward generic-rather-than-targeted
  features.

## Suggested splits

For a 10K balanced training set:
  - 4500 phishing (Nazario)
  - 4500 benign (Enron)
  - 500 spam (SpamAssassin) — boundary cases to keep precision honest
  - 500 phishing-style support / shipping notifications (synthetic) — to
    avoid the model learning "anything that looks transactional = phishing"

Hold out 1500 for validation, 1500 for test.

## Preprocessing

`src/utils.normalize_email` does most of the work:
  - drops headers (RFC 822 parsing)
  - decodes encoded-word subjects
  - replaces URLs, IPs, and email addresses with `<URL>`, `<IP>`, `<EMAIL>`
    so the model attends to *structural* features, not specific strings

There's a real tradeoff here. Concrete URL strings *do* contain signal —
attackers reuse infrastructure, so a model that memorizes "bank-secure-
login.example" would catch more attacks. But it also overfits to the train
set's specific phishing campaigns and degrades on novel ones. Neutralization
is the conservative choice; you can ablate it and see for yourself.

## Privacy

If you collect phishing reports from your own users:
  - never commit raw emails to a repo
  - hash sender addresses before retaining them
  - drop attachments
  - if your jurisdiction is GDPR-relevant, consult counsel before training

`.gitignore` excludes `data/` and `*.csv` for that reason. The
`examples/sample_emails.csv` file is synthetic — do not treat it as
real-world data.

## What goes wrong with this approach

- **Out-of-distribution phishing.** Highly targeted campaigns (spear-phishing,
  BEC) look like ordinary internal mail and don't trip the model.
- **Concept drift.** Attacker tactics shift. A model trained in 2019 against
  2018 data is already weaker; against 2020+ data it's worse. Plan for
  monthly retraining if this is a real deployment, not just a demo.
- **Adversarial inputs.** A phishing author who knows you classify with BERT
  can craft an email that looks benign to BERT (paraphrasing, different
  word distribution) but still tricks the recipient. Static text classifiers
  are not adversarial-robust by default.
