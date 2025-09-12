import argparse
import logging
import os
import json

from data.kr3_hf.load import load_kr3
from data.korean_webtext_hf.load import load_korean_webtext
from data.wikipedia_ko_hf.load import load_wikipedia_ko
from data.namuwiki_hf.load import load_namuwiki
from data.korean_textbook_hf.load import load_korean_textbook
from data.kullm_v2_hf.load import load_kullm_v2
from data.kmmlu_hf.load import load_kmmlu, load_category_kmmlu

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def clean_dataset(ds):
    return ds.filter(lambda x: isinstance(x.get("text"), str) and x["text"].strip() != "")

def save_lines_iter(dataset, path):
    n_written = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item.get("text")
            if not isinstance(text, str):
                continue
            line = " ".join(text.splitlines()).strip()
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

def save_jsonl(ds, output_dir, val_ratio=0.1, seed=0):
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)

    for key, suffix in [("train", "train_sft.jsonl"), ("test", "validation_sft.jsonl")]:
        file_path = os.path.join(output_dir, suffix)
        with open(file_path, "w", encoding="utf-8") as f:
            for ex in split[key]:
                prompt = ex["instruction"].strip()
                if ex.get("input", "").strip():
                    prompt += "\n\nInput: " + ex["input"].strip()
                json.dump({
                    "prompt": prompt,
                    "output": ex["output"].strip()
                }, f, ensure_ascii=False)
                f.write("\n")

        logging.info(f"Saved {len(split[key]):,} rows to {file_path}")

def save_kmmlu(ds, output_dir, split):
    path = os.path.join(output_dir, f"kmmlu_{split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            prompt = (
                f"\n\nQuestion: {ex['question']}\n"
                f"1. {ex['A']}\n"
                f"2. {ex['B']}\n"
                f"3. {ex['C']}\n"
                f"4. {ex['D']}\n"
            )
            answer = ex["answer"]
            json.dump({"prompt": prompt, "output": answer}, f, ensure_ascii=False)
            f.write("\n")

def save_kmmlu_category(ds, output_dir, split):
    supercats = set(ds["supercategory"])
    for supercat in supercats:
        filtered = ds.filter(lambda ex: ex["supercategory"] == supercat)
        path = os.path.join(output_dir, f"kmmlu_{supercat}_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in filtered:
                prompt = (
                    f"\n\nQuestion: {ex['question']}\n"
                    f"1. {ex['A']}\n"
                    f"2. {ex['B']}\n"
                    f"3. {ex['C']}\n"
                    f"4. {ex['D']}\n"
                )
                answer = ex["answer"]
                json.dump({"prompt": prompt, "output": answer}, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved {len(filtered)} rows to {path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == 'pre':
        ds_kr3 = load_kr3()
        ds_webtext = load_korean_webtext()
        ds_wiki = load_wikipedia_ko()
        ds_namu = load_namuwiki()
        ds_textbook = load_korean_textbook()

        split_and_save(ds_kr3, "kr3", args.output_dir, args.val_ratio)
        split_and_save(ds_webtext, "webtext", args.output_dir, args.val_ratio)
        split_and_save(ds_wiki, "wiki", args.output_dir, args.val_ratio)
        split_and_save(ds_namu, "namu", args.output_dir, args.val_ratio)
        split_and_save(ds_textbook, "textbook", args.output_dir, args.val_ratio)

        names = ["kr3", "webtext", "wiki", "namu", "textbook"]
        train_files = [os.path.join(args.output_dir, f"{name}_train.txt") for name in names]
        val_files =   [os.path.join(args.output_dir, f"{name}_val.txt")   for name in names]
        merge_files(train_files, os.path.join(args.output_dir, "train.txt"))
        merge_files(val_files, os.path.join(args.output_dir, "validation.txt"))

    elif args.mode == 'post':
        ds_kullm = load_kullm_v2()
        save_jsonl(ds_kullm, args.output_dir, val_ratio=args.val_ratio)

    elif args.mode == 'kmmlu':
        for split in ["train", "test"]:
            ds_kmmlu = load_kmmlu(split)
            save_kmmlu(ds_kmmlu, args.output_dir, split)

            ds_kmmlu = load_category_kmmlu(split)
            save_kmmlu_category(ds_kmmlu, args.output_dir, split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='pre', choices=['pre', 'post', 'kmmlu'])
    parser.add_argument('--output_dir', type=str, default="data")
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
