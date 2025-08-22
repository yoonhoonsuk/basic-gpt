import argparse, os, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import re
from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT
from tqdm import tqdm

def get_dataset(split="validation", data_dir="data"):
    path = os.path.join(data_dir, f"{split}_tokenized.pt")
    if os.path.exists(path):
        obj = torch.load(path)
        x, y, vocab_size = obj["x"], obj["y"], obj["vocab_size"]
        return TensorDataset(x, y), vocab_size
    raise FileNotFoundError(f"{path} not found. Run tokenization.py first.")

def get_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def parse_filename(path: str):
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

@torch.no_grad()
def validate(model, dataloader, criterion, device, n_tokens=None): 
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Validating", unit="batch")
    total_seen_tokens = 0 
    token_limit = n_tokens * 1_000_000 if n_tokens is not None else None 

    for idx, (x, y) in enumerate(pbar, 1):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        total_loss += loss.item()
        avg_loss = total_loss / idx
        pbar.set_postfix(loss=avg_loss)

        batch_tokens = x.numel()
        total_seen_tokens += batch_tokens 
        if token_limit is not None and total_seen_tokens >= token_limit: 
            print(f"Validation token limit {token_limit:,} reached. Stopping early.")
            break 

    avg_loss = total_loss / idx
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()

def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenized validation set
    dataset, vocab_size = get_dataset(split="validation", data_dir=args.data_dir)
    dataloader = get_data_loader(dataset, args.batch_size)

    state = torch.load(args.model_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    if args.arch == "decoder":
        ModelCls = GPT
    elif args.arch == "ffn":
        ModelCls = GPTFFNN
    else:
        is_decoder = any(("attn." in k or "pos_emb." in k) for k in state.keys())
        ModelCls = GPT if is_decoder else GPTFFNN

    # Load model
    model = ModelCls(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        max_len=args.chunk_size
    ).to(device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    # Validate
    avg_loss, ppl = validate(model, dataloader, criterion, device, n_tokens=args.n_tokens)
    print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--model_path", type=str, default="output/gpt_model_ddp.pt")
    parser.add_argument("--arch", type=str, choices=["ffn", "decoder"], default=None, 
                        help="Force model type; if omitted, auto-detect from checkpoint.")
    parser.add_argument("--n_tokens", type=int, default=None,
                        help="Limit validation to this many million tokens (e.g., 10=10M tokens).")
    args = parser.parse_args()

    parsed = parse_filename(args.model_path)
    if parsed:
        args.n_layers = parsed["n_layers"]
        args.d_model  = parsed["d_model"]
        args.d_ffn    = parsed["d_ffn"]
        print(f"[validate] Parsed from filename: "
              f"E{parsed['epochs']} N{args.n_layers} d_model={args.d_model} d_ffn={args.d_ffn}")

    main(args)
