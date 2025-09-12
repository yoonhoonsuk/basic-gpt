import argparse, os, torch, re
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT

from util.file_utils import parse_filename
from util.data_utils import get_dataset

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

    parsed = parse_filename(args.model_path)
    if parsed:
        args.n_layers = parsed["n_layers"]
        args.d_model  = parsed["d_model"]
        args.d_ffn    = parsed["d_ffn"]
        is_post = parsed.get("is_post", False)
    else:
        is_post = False

    dataset, vocab_size = get_dataset(
        split="validation",
        data_dir=args.data_dir,
        max_tokens=None,
        chunk_size=args.chunk_size,
        is_post=is_post
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

    model = ModelCls(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        max_len=args.chunk_size
    ).to(device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

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
    main(args)
