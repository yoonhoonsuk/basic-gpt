import argparse
import logging
import os
import json
import torch
from tqdm import tqdm

from preprocess.tokenizer_bbpe import tokenizer_bbpe, get_tokenizer, get_input_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def process_txt(input_txt, output_pt, chunk_size, batch_size):
    all_chunks = []
    vocab_size = None
    with open(input_txt, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    num_batches = (total_lines + batch_size - 1) // batch_size
    def stream_lines(filename, batch_size):
        batch = []
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch
    for batch_lines in tqdm(stream_lines(input_txt, batch_size=batch_size), total=num_batches, desc=f"Tokenizing {os.path.basename(input_txt)}"):
        chunks, vocab_size_ = tokenizer_bbpe(batch_lines, chunk_size=chunk_size)
        all_chunks.extend(chunks)
        vocab_size = vocab_size_
    arr = torch.tensor([t for chunk in all_chunks for t in chunk], dtype=torch.long).reshape(len(all_chunks), chunk_size+1)
    x = arr[:, :-1].contiguous()
    y = arr[:, 1:].contiguous()
    torch.save({"x": x, "y": y, "vocab_size": vocab_size}, output_pt)
    logging.info(f"Saved tokenized data to: {output_pt} (chunks: {len(all_chunks)})")

def stream_jsonl(fn, batch_size=512):
    batch = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            output = obj.get("output", "")
            batch.append((prompt, output))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def process_kmmlu(input_jsonl, output_pt, chunk_size=256, batch_size=512, is_test=False):
    from tqdm import tqdm
    import torch
    import logging
    from preprocess.tokenizer_bbpe import get_tokenizer, get_input_ids

    x_list, y_list = [], []
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    pad_id = 0  # HARD-CODE for simplicity; works for any reasonable vocab

    for batch in tqdm(stream_jsonl(input_jsonl, batch_size), desc="Tokenizing KMMLU"):
        for prompt, output in batch:
            input_text = prompt + "\n\n### Response:\n"
            target_text = str(output).strip()

            # Tokenize and OOV-safe
            x_ids = get_input_ids(tokenizer, [input_text])[0]
            y_ids = get_input_ids(tokenizer, [target_text])[0]
            x_ids = [t if 0 <= t < vocab_size else 0 for t in x_ids]
            y_ids = [t if 0 <= t < vocab_size else 0 for t in y_ids]

            # Pad or truncate to chunk_size
            if len(x_ids) > chunk_size:
                x_ids = x_ids[-chunk_size:]
            else:
                x_ids = [pad_id] * (chunk_size - len(x_ids)) + x_ids
            x = torch.tensor(x_ids, dtype=torch.long)

            # Target: only answer at last position, rest masked
            y = torch.full((chunk_size,), -100, dtype=torch.long)
            if y_ids:
                y[-1] = y_ids[0]
            else:
                continue  # skip broken sample

            # Sanity check for OOV (REMOVE this in real pipeline)
            assert (x.numpy() >= 0).all() and (x.numpy() < vocab_size).all(), "x OOV!"
            assert y[-1] < vocab_size and y[-1] >= 0, "y OOV!"

            x_list.append(x)
            y_list.append(y)

    assert x_list, "No valid samples found!"
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    torch.save({"x": x, "y": y, "vocab_size": vocab_size, "pad_id": pad_id}, output_pt)
    print(f"[INFO] Saved tokenized data to: {output_pt} (chunks: {len(x_list)})")


def process_jsonl(input_jsonl, output_pt, chunk_size=256, batch_size=512):
    x_list, y_list = [], []
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    window_len = chunk_size + 1
    skipped, total, oov_count = 0, 0, 0
    for batch in tqdm(stream_jsonl(input_jsonl, batch_size), desc=f"Tokenizing {os.path.basename(input_jsonl)}"):
        texts, mask_prefixes = [], []
        for prompt, output in batch:
            texts.append(prompt + "\n\n### Response:\n" + str(output))
            mask_prefixes.append(prompt + "\n\n### Response:\n")
        input_ids_list = get_input_ids(tokenizer, texts)
        mask_prefix_ids_list = get_input_ids(tokenizer, mask_prefixes)
        for token_ids, mask_prefix_ids in zip(input_ids_list, mask_prefix_ids_list):
            if not token_ids:
                skipped += 1
                continue
            prompt_len = len(mask_prefix_ids)
            n_chunks = (len(token_ids) + chunk_size) // chunk_size
            for i in range(n_chunks):
                raw_chunk = token_ids[i*chunk_size : i*chunk_size + window_len]
                oov_count += sum(1 for t in raw_chunk if not (0 <= t < vocab_size))
                if len(raw_chunk) < window_len:
                    raw_chunk = raw_chunk + [pad_id] * (window_len - len(raw_chunk))
                chunk = [t if 0 <= t < vocab_size else 0 for t in raw_chunk]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                start_pos = i * chunk_size
                mask_len = max(0, min(prompt_len - 1 - start_pos, y.size(0)))
                if mask_len > 0:
                    y[:mask_len] = -100
                y[y == pad_id] = -100
                if not (y >= 0).any():
                    skipped += 1
                    continue
                x_list.append(x)
                y_list.append(y)
                total += 1
    if not x_list:
        raise RuntimeError("No samples could be processed successfully.")
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    torch.save({"x": x, "y": y, "vocab_size": vocab_size, "pad_id": pad_id}, output_pt)
    logging.info(f"Saved {total} chunks, skipped {skipped}, OOV tokens seen: {oov_count}")
    logging.info(f"Shapes: x={tuple(x.shape)}, y={tuple(y.shape)}, vocab_size={vocab_size}")

def main(args):
    filemap = {
        "pre": [
            ("train.txt", "train_tokenized.pt"),
            ("validation.txt", "validation_tokenized.pt"),
        ],
        "post": [
            ("train_sft.jsonl", "train_sft_tokenized.pt"),
            ("validation_sft.jsonl", "validation_sft_tokenized.pt"),
        ],
        "kmmlu": [
            ("kmmlu_train.jsonl", "kmmlu_train_tokenized.pt"),

            ("kmmlu_Applied Science_test.jsonl", "kmmlu_Applied Science_test_tokenized.pt"),
            ("kmmlu_HUMSS_test.jsonl", "kmmlu_HUMSS_test_tokenized.pt"),
            ("kmmlu_Other_test.jsonl", "kmmlu_Other_test_tokenized.pt"),
            ("kmmlu_STEM_test.jsonl", "kmmlu_STEM_test_tokenized.pt"),
            ("kmmlu_test.jsonl", "kmmlu_test_tokenized.pt")
        ],
    }

    for input_file, output_file in filemap[args.mode]:
        input_path = os.path.join(args.data_dir, input_file)
        output_path = os.path.join(args.data_dir, output_file)

        if not os.path.exists(input_path):
            logging.warning(f"File not found: {input_path} (skipping)")
            continue
        if input_file.endswith(".txt"):
            process_txt(input_path, output_path, chunk_size=args.chunk_size, batch_size=args.batch_size)
        elif "kmmlu" in input_file:
            is_test = "test" in input_file
            process_kmmlu(input_path, output_path, chunk_size=args.chunk_size, batch_size=args.batch_size, is_test=is_test)
        else:            
            process_jsonl(input_path, output_path, chunk_size=args.chunk_size, batch_size=args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--mode", type=str, required=True, choices=["pre", "post", "kmmlu"])
    args = parser.parse_args()
    main(args)
