import argparse
import logging
import os
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def count_lines(filename):
    # Efficient line count for tqdm total
    with open(filename, encoding="utf-8") as f:
        for i, _ in enumerate(f, 1):
            pass
    return i

def stream_tokenize_file(input_path, tokenizer, chunk_size, batch_size, output_path):
    total_lines = count_lines(input_path)
    logging.info(f"Tokenizing {input_path} in batches of {batch_size}...")
    all_ids = []
    batch = []
    with open(input_path, encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc=os.path.basename(input_path)):
            line = line.strip()
            if line:
                batch.append(line)
                if len(batch) >= batch_size:
                    ids = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
                    all_ids.extend(id for ids_ in ids if len(ids_) > 1 for id in ids_)
                    batch = []
        if batch:
            ids = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
            all_ids.extend(id for ids_ in ids if len(ids_) > 1 for id in ids_)

    all_ids = np.array(all_ids, dtype=np.int64)
    num_chunks = len(all_ids) // chunk_size
    if num_chunks == 0:
        raise ValueError(f"Not enough tokens for chunk_size={chunk_size} in {input_path}")
    all_ids = all_ids[:num_chunks * chunk_size]
    arr = torch.tensor(all_ids, dtype=torch.long).reshape(num_chunks, chunk_size)
    x = arr[:, :-1].contiguous()
    y = arr[:, 1:].contiguous()
    torch.save({"x": x, "y": y, "vocab_size": tokenizer.vocab_size}, output_path)
    logging.info(f"Saved tokenized data to: {output_path} (chunks: {num_chunks})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_name", type=str, default="skt/kogpt2-base-v2")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Tokenize train.txt
    train_txt = os.path.join(args.data_dir, "train.txt")
    train_out = os.path.join(args.data_dir, "train_tokenized.pt")
    stream_tokenize_file(train_txt, tokenizer, args.chunk_size, args.batch_size, train_out)

    # Tokenize validation.txt
    val_txt = os.path.join(args.data_dir, "validation.txt")
    val_out = os.path.join(args.data_dir, "validation_tokenized.pt")
    stream_tokenize_file(val_txt, tokenizer, args.chunk_size, args.batch_size, val_out)
