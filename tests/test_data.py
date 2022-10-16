"""Tests for src.data — EmailDataset + stratified split."""
import csv
import os
import tempfile
import unittest

from data import EmailDataset, stratified_split


class FakeTokenizer:
    """Minimal tokenizer that mimics BertTokenizer's encode_plus return shape."""
    def encode_plus(self, text, **kwargs):
        max_len = kwargs.get('max_length', 16)
        ids = [101] + [hash(w) % 30000 for w in text.split()][:max_len - 2] + [102]
        ids += [0] * (max_len - len(ids))
        mask = [1 if i != 0 else 0 for i in ids]
        return {
            'input_ids': [ids],
            'attention_mask': [mask],
        }


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['text', 'label'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestEmailDataset(unittest.TestCase):
    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'a.csv')
            _write_csv([
                {'text': 'click here now', 'label': '1'},
                {'text': 'project update', 'label': '0'},
            ], path)
            ds = EmailDataset(path, FakeTokenizer(), max_len=16)
            self.assertEqual(len(ds), 2)
            item = ds[0]
            self.assertIn('input_ids', item)
            self.assertIn('attention_mask', item)
            self.assertIn('label', item)

    def test_skips_empty_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'a.csv')
            _write_csv([
                {'text': '', 'label': '1'},
                {'text': 'hi', 'label': '0'},
            ], path)
            ds = EmailDataset(path, FakeTokenizer(), max_len=16)
            self.assertEqual(len(ds), 1)


class TestStratifiedSplit(unittest.TestCase):
    def test_proportions_preserved(self):
        labels = [0]*80 + [1]*20
        train, val = stratified_split(labels, val_frac=0.2, seed=0)
        # Should preserve 4:1 ratio in both partitions
        train_l = [labels[i] for i in train]
        val_l = [labels[i] for i in val]
        self.assertAlmostEqual(train_l.count(1) / len(train_l), 0.2, places=1)
        self.assertAlmostEqual(val_l.count(1) / len(val_l), 0.2, places=1)


if __name__ == '__main__':
    unittest.main()
