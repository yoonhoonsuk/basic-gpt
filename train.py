import os
import argparse
import time
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from preprocess.manual_bbpe import manual_bbpe
from preprocess.tokenizer_bbpe import tokenizer_bbpe
from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT

warnings.filterwarnings("ignore", category=FutureWarning)

def get_dataset(split="train", data_dir="data"):
    path = os.path.join(data_dir, f"{split}_tokenized.pt")
    if os.path.exists(path):
        obj = torch.load(path)
        x, y, vocab_size = obj["x"], obj["y"], obj["vocab_size"]

        total_tokens = x.numel()   # total elements = batch * seq_len
        print(f"{split} set: {total_tokens:,} tokens")

        return TensorDataset(x, y), vocab_size
    raise FileNotFoundError(f"{path} not found. Run tokenization first.")

def get_data_loader(dataset, batch_size):
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train(model, dataloader, optimizer, criterion, device, tokens_left=None):
    model.train()
    total_loss = 0.0
    scaler = GradScaler() 
    progress = tqdm(dataloader, desc="Batch", leave=False)
    trained_tokens = 0
    for x, y in progress:
        if tokens_left is not None and trained_tokens >= tokens_left:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast() :
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

        batch_tokens = x.numel()
        trained_tokens += batch_tokens
    return total_loss / len(dataloader), trained_tokens

def pick_nhead(d_model: int, nhead_arg: int = None, target_head_dim: int = 64) -> int:
    if nhead_arg is not None:
        assert d_model % nhead_arg == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead_arg})"
        return nhead_arg
    h = max(1, round(d_model / target_head_dim))
    while h > 1 and d_model % h != 0:
        h -= 1
    return h

def main(args):
    # Distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and vocab_size from tokenized .pt
    dataset, vocab_size = get_dataset(split="train", data_dir=args.data_dir)
    dataloader = get_data_loader(dataset, args.batch_size)

    nhead = pick_nhead(args.d_model, args.nhead)

    # Model
    if args.arch == "ffn":
        model = GPTFFNN(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_ffn=args.d_ffn,
            n_layers=args.n_layers,
            max_len=args.chunk_size,
            dropout=args.dropout,
        ).to(device)
    else:
        model = GPT(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_ffn=args.d_ffn,
            n_layers=args.n_layers,
            max_len=args.chunk_size,
            dropout=args.dropout,
            nhead=nhead,
        ).to(device)

    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    n_tokens = args.n_tokens * 1_000_000 if args.n_tokens is not None else None
    total_trained_tokens = 0

    start_time = time.time()
    for epoch in range(args.epochs):
        if dist.is_available() and dist.is_initialized():
            dataloader.sampler.set_epoch(epoch)

        tokens_left = n_tokens
        loss, epoch_tokens = train(model, dataloader, optimizer, criterion, device, tokens_left=tokens_left)
        total_trained_tokens += epoch_tokens

        if local_rank == 0:
            print(f"Loss: {loss:.4f} | Tokens trained in epoch: {epoch_tokens:,} | Total tokens: {total_trained_tokens:,}")
        scheduler.step()

    if local_rank == 0:
        print(f"Training took {time.time() - start_time:.2f} seconds.")
        filename = f"E{args.epochs}_N{args.n_layers}_{args.d_model}_{args.d_ffn}_{loss:.4f}.pt"
        os.makedirs("output", exist_ok=True)
        save_path = os.path.join("output", filename)
        torch.save(model.module.state_dict() if dist.is_initialized() else model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="decoder", choices=["ffn", "decoder"],
                        help="Choose 'ffn' (MLP-only) or 'decoder' (Transformer).")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="tokenizer", choices=["tokenizer", "manual"])
    parser.add_argument("--nhead", type=int, default=None, help="Number of attention heads (defaults to ~d_model/64).")
    parser.add_argument("--n_tokens", type=int, default=30, help="Train up to this many million tokens PER EPOCH (e.g., 300 = 300M tokens per epoch). If not set, train full dataset each epoch.")
    args = parser.parse_args()
    main(args)
