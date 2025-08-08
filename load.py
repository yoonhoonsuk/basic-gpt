import argparse
import logging
import os
import torch
from data.kr3_hf.load import load_kr3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def prepare_dataset(output_path, val_ratio=0.05):
    base_dir = os.path.dirname(output_path) or "."
    os.makedirs(base_dir, exist_ok=True)

    ds = load_kr3(split="train")
    split_ds = ds.train_test_split(test_size=val_ratio)

    torch.save({"texts": list(split_ds["train"]["text"])}, os.path.join(base_dir, "train.pt"))
    torch.save({"texts": list(split_ds["test"]["text"])}, os.path.join(base_dir, "validation.pt"))

    logging.info(f"Train samples: {len(split_ds['train'])}")
    logging.info(f"Validation samples: {len(split_ds['test'])}")
    logging.info(f"Saved to {base_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", type=str, default="data/dataset.pt")
    p.add_argument("--val_ratio", type=float, default=0.05)
    args = p.parse_args()
    prepare_dataset(args.output_path, args.val_ratio)
