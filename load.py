import argparse
import logging
import os
from datasets import concatenate_datasets

from data.kr3_hf.load import load_kr3
from data.korean_webtext_hf.load import load_korean_webtext
from data.wikipedia_ko_hf.load import load_wikipedia_ko
from data.namuwiki_hf.load import load_namuwiki
from data.korean_textbook_hf.load import load_korean_textbook

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def clean_dataset(ds):
    # Keep rows where text is a non-empty string after stripping
    return ds.filter(lambda x: isinstance(x.get("text"), str) and x["text"].strip() != "")

def save_lines_iter(dataset, path):
    n_written = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item.get("text")
            if not isinstance(text, str):
                continue
            line = " ".join(text.splitlines()).strip()  # remove newlines; collapse whitespace edges
            if not line:
                continue
            f.write(line + "\n")
            n_written += 1
    logging.info(f"Wrote {n_written:,} lines to {path}")

def split_and_save(ds, name, output_dir, val_ratio=0.1, seed=0):
    ds = clean_dataset(ds)
    split = ds.shuffle(seed=seed).train_test_split(test_size=val_ratio, seed=seed)
    save_lines_iter(split["train"], os.path.join(output_dir, f"{name}_train.txt"))
    save_lines_iter(split["test"], os.path.join(output_dir, f"{name}_val.txt"))

def merge_files(files, output_path):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for fname in files:
            with open(fname, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

def main(output_dir, val_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    # Load datasets
    ds_kr3 = load_kr3()
    ds_webtext = load_korean_webtext()
    ds_wiki = load_wikipedia_ko()
    ds_namu = load_namuwiki()
    ds_textbook = load_korean_textbook()

    # Split and save each dataset
    split_and_save(ds_kr3, "kr3", output_dir, val_ratio)
    split_and_save(ds_webtext, "webtext", output_dir, val_ratio)
    split_and_save(ds_wiki, "wiki", output_dir, val_ratio)
    split_and_save(ds_namu, "namu", output_dir, val_ratio)
    split_and_save(ds_textbook, "textbook", output_dir, val_ratio)

    # Merge splits for train and validation
    names = ["kr3", "webtext", "wiki", "namu", "textbook"]
    train_files = [os.path.join(output_dir, f"{name}_train.txt") for name in names]
    val_files =   [os.path.join(output_dir, f"{name}_val.txt")   for name in names]
    merge_files(train_files, os.path.join(output_dir, "train.txt"))
    merge_files(val_files, os.path.join(output_dir, "validation.txt"))
    logging.info("Merged train/validation files.")

if __name__ == "__main__":
    main("data", val_ratio=0.1)
