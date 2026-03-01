"""
HC3 dataset loader, tokenization and preprocessing.
Uses BertTokenizer only as a tool for WordPiece tokenization — no pretrained model weights.

Expected file: data/hc3_all.jsonl
Download from: https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl
"""

import json
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer


# constants

MAX_LEN = 256  # max token sequence length (use 512 for final training!!!)
TOKENIZER_NAME = "bert-base-uncased"
DATA_PATH = "data/hc3_all.jsonl"


# loading and parsing HC3

def load_hc3(path: str = DATA_PATH, test_split: float = 0.1, seed: int = 9) -> tuple:
    """
    Reads HC3 from a local .jsonl file and returns two flat lists of samples.

    HC3 row structure:
        {
          "question": str,
          "human_answers":  [str, ...],   -> label 0
          "chatgpt_answers": [str, ...]   -> label 1
        }

    Splits rows (not samples) into train/test before flattening answers,
    so there is no question-level leakage between splits.

    Args:
        path:       path to hc3_all.jsonl
        test_split: fraction of rows reserved for test
        seed:       random seed for reproducible split

    Returns:
        (train_samples, test_samples) — each a list of {"text": str, "label": int}
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # shuffle rows before splitting
    rng = random.Random(seed)
    rng.shuffle(rows)

    split_idx = int(len(rows) * (1 - test_split))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    return _rows_to_samples(train_rows), _rows_to_samples(test_rows)


def _rows_to_samples(rows: list) -> list:
    """Flattens a list of HC3 rows into individual {text, label} samples."""
    samples = []
    for row in rows:
        for answer in row.get("human_answers", []):
            text = _clean(answer)
            if text:
                samples.append({"text": text, "label": 0})
        for answer in row.get("chatgpt_answers", []):
            text = _clean(answer)
            if text:
                samples.append({"text": text, "label": 1})
    return samples


def _clean(text: str) -> str:
    """Strips whitespace and collapses multiple spaces."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


# dataset class

class HC3Dataset(Dataset):
    """
    PyTorch Dataset for HC3 binary classification.

    Tokenizes each text with BertTokenizer (WordPiece, vocab size 30,522).
    BertTokenizer automatically prepends [CLS] and appends [SEP].

    Returns per sample:
        input_ids      [max_len]  token indices (including [CLS] and [SEP])
        attention_mask [max_len]  1 for real tokens, 0 for [PAD]
        label          scalar     0 = human, 1 = AI

    The [CLS] token at position 0 is used as the whole-sequence representation.
    """

    def __init__(self, samples: list, tokenizer: BertTokenizer, max_len: int = MAX_LEN):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text = self.samples[idx]["text"]
        label = self.samples[idx]["label"]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",  # pad short sequences to max_len with [PAD]
            truncation=True,       # cut sequences longer than max_len
            return_tensors="pt"    # return PyTorch tensors
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),       # [max_len]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [max_len]
            "label": torch.tensor(label, dtype=torch.long)
        }


# collate function

def collate_fn(batch: list) -> dict:
    """
    Stacks individual sample dicts into batched tensors.
    Since we already pad to max_len in __getitem__, this is just torch.stack.
    """
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),           # [B, max_len]
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]), # [B, max_len]
        "labels": torch.stack([x["label"] for x in batch]),                  # [B]
    }


# main pipeline

def build_dataloaders(
    batch_size: int = 32,
    max_len: int = MAX_LEN,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,   # 0 = main process (safe on Windows)
    seed: int = 9,
    data_path: str = DATA_PATH
) -> tuple:
    """
    Full data pipeline:
      1. Load HC3 from local .jsonl and split into train/test rows
      2. Init BertTokenizer
      3. Build HC3Dataset objects
      4. Further split train into train/val for early stopping
      5. Return DataLoaders and tokenizer

    Split strategy:
        Rows are split into train/test BEFORE flattening answers,
        preventing any question from appearing in both splits.
        Val is carved out of train for early stopping monitoring.
        Test is never touched during training or hyperparameter tuning.

    Returns:
        (train_loader, val_loader, test_loader, tokenizer)
    """
    print("Loading HC3 dataset...")
    train_samples, test_samples = load_hc3(path=data_path, test_split=test_split, seed=seed)

    print(f"  Train+val samples: {len(train_samples)}")
    print(f"  Test samples     : {len(test_samples)}")

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    full_train_ds = HC3Dataset(train_samples, tokenizer, max_len)
    test_ds = HC3Dataset(test_samples, tokenizer, max_len)

    # split train into train/val
    n_val = int(len(full_train_ds) * val_split)
    n_train = len(full_train_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val], generator=generator)

    print(f"  Train            : {len(train_ds)}")
    print(f"  Val              : {len(val_ds)}")
    print(f"  Test             : {len(test_ds)}")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, tokenizer


# quick sanity check
if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = build_dataloaders(
        batch_size=4,
        max_len=128
    )

    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"  input_ids      : {batch['input_ids'].shape}")
    print(f"  attention_mask : {batch['attention_mask'].shape}")
    print(f"  labels         : {batch['labels']}")

    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    print(f"\nFirst sample (first 20 tokens): {tokens[:20]}")
    print(f"Label: {batch['labels'][0].item()}  (0=human, 1=AI)")
