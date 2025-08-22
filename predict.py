import sys, re, os, torch
import argparse
from transformers import AutoTokenizer

from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT

def load_model(model_path, vocab_size, d_model, d_ffn, n_layers, chunk_size, device, arch=None):  # <-- arch added
    state_dict = torch.load(model_path, map_location=device)

    # choose class: use flag if provided, else auto-detect by keys
    if arch == "decoder":
        ModelCls = GPT
    elif arch == "ffn":
        ModelCls = GPTFFNN
    else:
        keys = state_dict.keys() if isinstance(state_dict, dict) else []
        is_decoder = any(("attn." in k or "pos_emb" in k) for k in keys)
        ModelCls = GPT if is_decoder else GPTFFNN

    model = ModelCls(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ffn=d_ffn,
        n_layers=n_layers,
        max_len=chunk_size
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def parse_filename(path: str):
    """Parse E{epochs}_N{n_layers}_{d_model}_{d_ffn}.pt"""
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    m = re.match(r'^E(?P<E>\d+)_N(?P<N>\d+)_(?P<D>\d+)_(?P<F>\d+)', stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "epochs": int(g["E"]),
        "n_layers": int(g["N"]),
        "d_model": int(g["D"]),
        "d_ffn": int(g["F"]),
    }

def clean_prompt(prompt):
    return ''.join(ch for ch in prompt if ch.isprintable())

def generate(prompt, model, tokenizer, device, max_new_tokens=50, stop_tokens=(".", "!", "?", "\n")):
    try:
        prompt = clean_prompt(prompt)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    except Exception as e:
        print(f"[ERROR] Tokenizer error: {e}")
        return ""
    if "input_ids" not in encoded or encoded["input_ids"].numel() == 0:
        print("[WARNING] Prompt produced no tokens. Please enter a valid prompt.")
        return ""
    input_ids = encoded["input_ids"].to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        decoded_token = tokenizer.decode(next_token[0])
        if any(token in decoded_token for token in stop_tokens):
            break

    return tokenizer.decode(generated[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="output/gpt_model_ddp.pt")
    parser.add_argument('--arch', type=str, choices=['ffn', 'decoder'], default=None)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ffn', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--max_token', type=int, default=50)
    args = parser.parse_args()

    parsed = parse_filename(args.model_path)
    if parsed:
        args.n_layers = parsed["n_layers"]
        args.d_model  = parsed["d_model"]
        args.d_ffn    = parsed["d_ffn"]
        print(f"[info] Parsed from filename: "
              f"E{parsed['epochs']} N{args.n_layers} d_model={args.d_model} d_ffn={args.d_ffn}")

    try:
        tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
        vocab_size = tokenizer.vocab_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model(
            args.model_path, vocab_size, args.d_model, args.d_ffn,
            args.n_layers, args.chunk_size, device, arch=args.arch
        )
    except Exception as e:
        print(f"[ERROR] Model/tokenizer loading failed: {e}")
        sys.exit(1)

    while True:
        prompt = input("Prompt (or 'exit' to quit): ").strip()
        if not prompt or prompt.lower() == "exit":
            print("Exiting.")
            break
        try:
            output = generate(prompt, model, tokenizer, device, max_new_tokens=args.max_token)
            print(f"Generated: {output}\n")
        except Exception as e:
            print(f"[ERROR] During generation: {e}\n")

