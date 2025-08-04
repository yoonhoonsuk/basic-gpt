import argparse
import logging
import torch
from data.kr3_hf.load import load_kr3

def prepare_dataset(split: str, output_path: str) -> None:
    """
    Loads and cleans the KR3 dataset, then saves to disk.
    """
    dataset = load_kr3(split=split)  # already filters ratings, renames column

    texts = list(dataset["text"])
    logging.info(f"Loaded {len(texts):,} text samples.")

    torch.save({"texts": texts}, output_path)
    logging.info(f"Saved cleaned dataset to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare KR3 dataset for training.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--output_path", type=str, default="data/dataset.pt", help="Where to save the cleaned dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.split, args.output_path)
