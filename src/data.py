"""Email dataset loader. Expects CSV with `text` and `label` columns."""
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset


class EmailDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=256, drop_empty=True):
        df = pd.read_csv(csv_path)
        if drop_empty:
            df = df[df['text'].astype(str).str.strip().str.len() > 0]
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            pad_to_max_length=True,  # the 2019 kwarg
            truncation=True,
            return_tensors='pt',
        )
        ids = enc['input_ids']
        mask = enc['attention_mask']
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)
        return {
            'input_ids': ids.squeeze(0) if ids.dim() == 2 else ids,
            'attention_mask': mask.squeeze(0) if mask.dim() == 2 else mask,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


