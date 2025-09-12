import os
import torch
from torch.utils.data import TensorDataset

def get_dataset(split="train", data_dir="data", max_tokens=None, chunk_size=256, is_post=False):
    if is_post:
        # path = os.path.join(data_dir, f"{split}_sft_tokenized.pt")
        path = os.path.join(data_dir, f"kmmlu_{split}_tokenized.pt")
        # path = os.path.join(data_dir, f"kmmlu_test_tokenized.pt")
    else:
        path = os.path.join(data_dir, f"{split}_tokenized.pt")

    if os.path.exists(path):
        obj = torch.load(path)
        x, y, vocab_size = obj["x"], obj["y"], obj["vocab_size"]
        if max_tokens is not None:
            num_samples = max_tokens // chunk_size
            x = x[:num_samples]
            y = y[:num_samples]
        num_tokens = x.numel()
        print(f"Loaded {num_tokens:,} tokens from {path} (shape: {x.shape})")
        return TensorDataset(x, y), vocab_size
    raise FileNotFoundError(f"{path} not found. Run tokenization.py first.")
