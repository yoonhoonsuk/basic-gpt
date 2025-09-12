import os
import argparse
import torch
import joblib
from preprocess.tokenizer_bbpe import get_tokenizer, get_input_ids
from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT
from util.file_utils import parse_filename

def load_model(args, vocab_size, device):
    parsed = parse_filename(args.model_path)
    if parsed:
        args.n_layers = parsed["n_layers"]
        args.d_model = parsed["d_model"]
        args.d_ffn = parsed["d_ffn"]
    arch = args.arch or ("ffn" if "ffn" in args.model_path.lower() else "decoder")
    model_max_len = args.chunk_size
    if arch == "decoder":
        model = GPT(vocab_size=vocab_size, d_model=args.d_model, d_ffn=args.d_ffn, n_layers=args.n_layers, max_len=model_max_len).to(device)
    elif arch == "ffn":
        model = GPTFFNN(vocab_size=vocab_size, d_model=args.d_model, d_ffn=args.d_ffn, n_layers=args.n_layers, max_len=model_max_len).to(device)
    else:
        raise ValueError("arch must be 'ffn' or 'decoder'")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def sentiment_loop(args, tokenizer, device):
    vocab_size = tokenizer.vocab_size
    model = load_model(args, vocab_size, device)
    probe = joblib.load(args.probe_pkl)
    clf, scaler = probe["probe"], probe["scaler"]
    print("[INFO] Sentiment mode. Type input (empty line or 'exit' to quit):")
    while True:
        prompt = input("Prompt: ").strip()
        if not prompt or prompt.lower() == "exit":
            print("Exiting.")
            break
        try:
            ids = get_input_ids(tokenizer, [prompt])[0][:args.chunk_size]
            if len(ids) < args.chunk_size:
                ids += [0] * (args.chunk_size - len(ids))
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                _, feats = model(x, return_hidden=True)
                feats = feats[:, -1, :].cpu().numpy()
            feats_scaled = scaler.transform(feats)
            pred = clf.predict(feats_scaled)[0]
            proba = clf.predict_proba(feats_scaled)[0]
            label = "POS" if pred == 1 else "NEG"
            pos_prob = proba[1] if len(proba) > 1 else float(proba)
            print(f"Prediction: {label}  P(POS)={pos_prob:.3f}  P(NEG)={proba[0]:.3f}\n")
        except Exception as e:
            print(f"[ERROR] {e}")

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., -1, None]] = -float('Inf')
    return out

def format_prompt(text, is_post):
    return text.strip() + "\n\n### Response:\n" if is_post else text.strip()

def generate(text, model, tokenizer, device, chunk_size, max_token=50, k=20, is_post=False):
    prompt = format_prompt(text, is_post)
    ids = get_input_ids(tokenizer, [prompt])[0]
    if not ids:
        raise ValueError(f"No tokens produced from prompt: '{text}'")
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_token):
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            logits = top_k_logits(logits, k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)

def gen_file(args, tokenizer, device):
    vocab_size = tokenizer.vocab_size
    is_post = "_post" in args.model_path.lower()
    model = load_model(args, vocab_size, device)
    if is_post:
        print("[INFO] Running in post-trained (SFT) mode.")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(args.model_path))[0]
    out_path = os.path.join(out_dir, f"{fname}_output.txt")

    with open(args.input_file, encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, prompt in enumerate(prompts, 1):
            try:
                output = generate(prompt, model, tokenizer, device, args.chunk_size, args.max_token, args.top_k, is_post)
            except Exception as e:
                output = f"[ERROR] {e}"
            fout.write(f"[{idx}] Prompt: {prompt}\n[{idx}] Output: {output}\n\n")
    print(f"[INFO] Results written to {out_path}")

def gen_loop(args, tokenizer, device):
    vocab_size = tokenizer.vocab_size
    is_post = "_post" in args.model_path.lower()
    model = load_model(args, vocab_size, device)
    print("[INFO] Generation mode. Type input (empty line or 'exit' to quit):")
    while True:
        prompt = input("Prompt: ").strip()
        if not prompt or prompt.lower() == "exit":
            print("Exiting.")
            break
        try:
            output = generate(prompt, model, tokenizer, device, args.chunk_size, args.max_token, args.top_k, is_post)
            print(f"Output: {output}\n")
        except Exception as e:
            print(f"[ERROR] {e}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    if args.input_file:
        if args.probe_pkl:
            raise ValueError("--input_file and --probe_pkl cannot be used together.")
        gen_file(args, tokenizer, device)
        return

    if args.probe_pkl:
        sentiment_loop(args, tokenizer, device)
    else:
        gen_loop(args, tokenizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to model .pt file (required)")
    parser.add_argument('--probe_pkl', type=str, default=None, help='Pickle file with probe+scaler (for sentiment)')
    parser.add_argument('--arch', type=str, choices=['ffn', 'decoder'], default=None)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ffn', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--chunk_size', type=int, default=256, help='Should match training chunk_size')
    parser.add_argument('--input_file', type=str, default=None, help='Input .txt file for batch generation')
    parser.add_argument('--max_token', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k sampling for generation')
    args = parser.parse_args()
    main(args)
