import torch
import argparse
from preprocess.tokenizer_bbpe import get_tokenizer

def analyze_sample(x, y, tokenizer, pad_id=0):
    # Remove padding from x for readable prompt
    x_list = x.tolist()
    y_list = y.tolist()

    # Print prompt (ignore padding)
    prompt_tokens = [t for t in x_list if t != pad_id]
    prompt_text = tokenizer.decode(prompt_tokens)

    # Print answer (where y != -100)
    answer_tokens = [t for t, yy in zip(y_list, y_list) if yy != -100 and t != pad_id]
    answer_text = tokenizer.decode(answer_tokens)

    # Where is y unmasked?
    unmasked_pos = [i for i, yy in enumerate(y_list) if yy != -100]

    print("Decoded prompt (x):")
    print(repr(prompt_text))
    print("Decoded answer (unmasked y):")
    print(repr(answer_text))
    print(f"Unmasked positions in y: {unmasked_pos}")
    print(f"Prompt length (non-pad in x): {len(prompt_tokens)}")
    print(f"Answer length (unmasked in y): {len(answer_tokens)}")
    print("-" * 60)

def main(args):
    tokenizer = get_tokenizer()
    print(f"[INFO] Loading tokenized data from {args.pt_path}")
    obj = torch.load(args.pt_path, map_location="cpu")
    x = obj["x"]
    y = obj["y"]
    pad_id = obj.get("pad_id", 0)

    print(f"[INFO] x shape: {x.shape}, y shape: {y.shape}, pad_id: {pad_id}")
    print("=" * 60)
    for i in range(args.num_samples):
        print(f"[Sample {i+1}]")
        analyze_sample(x[i], y[i], tokenizer, pad_id=pad_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    main(args)
