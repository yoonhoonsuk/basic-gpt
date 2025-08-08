import sys
import argparse
import torch
from model.FFNN import GPTFFNN
from transformers import AutoTokenizer

def load_model(model_path, vocab_size, d_model, d_ffn, n_layers, chunk_size, device):
    model = GPTFFNN(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ffn=d_ffn,
        n_layers=n_layers,
        max_len=chunk_size
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

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
    parser.add_argument('--model_path', type=str, default="output/E72_N6_384_4.5747.pt")
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ffn', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--max_token', type=int, default=50)
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
        vocab_size = tokenizer.vocab_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.model_path, vocab_size, args.d_model, args.d_ffn, args.n_layers, args.chunk_size, device)
    except Exception as e:
        print(f"[ERROR] Model/tokenizer loading failed: {e}")
        sys.exit(1)

    while True:
        prompt = input("Prompt (or 'exit' to quit): ").strip()
        
        if not prompt or prompt.lower() == "exit":
            print("Exiting.")
            break

        # print("prompt:", repr(prompt))
        # print("tokenizer.tokenize(prompt):", tokenizer.tokenize(prompt))
        # print("tokenizer.encode(prompt):", tokenizer.encode(prompt))

        try:
            output = generate(prompt, model, tokenizer, device, max_new_tokens = args.max_token)
            print(f"Generated: {output}\n")
        except Exception as e:
            print(f"[ERROR] During generation: {e}\n")



