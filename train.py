import os
import argparse
import math
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from preprocess.manual_bbpe import manual_bbpe
from preprocess.tokenizer_bbpe import tokenizer_bbpe
from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT

from util.file_utils import parse_filename
from util.data_utils import get_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

def get_data_loader(dataset, batch_size):
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_device():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return local_rank, device

def get_model(arch, vocab_size, d_model, d_ffn, n_layers, chunk_size, dropout, nhead, device, use_ddp=True, local_rank=0):
    if arch == "ffn":
        model = GPTFFNN(vocab_size=vocab_size, d_model=d_model, d_ffn=d_ffn, n_layers=n_layers, max_len=chunk_size, dropout=dropout).to(device)
    else:
        model = GPT(vocab_size=vocab_size, d_model=d_model, d_ffn=d_ffn, n_layers=n_layers, max_len=chunk_size, dropout=dropout, nhead=nhead).to(device)
    if use_ddp and dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    return model

def pick_nhead(d_model: int, nhead_arg: int = None, target_head_dim: int = 64) -> int:
    if nhead_arg is not None:
        assert d_model % nhead_arg == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead_arg})"
        return nhead_arg
    h = max(1, round(d_model / target_head_dim))
    while h > 1 and d_model % h != 0:
        h -= 1
    return h

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0
    scaler = GradScaler()
    progress = tqdm(dataloader, desc="Batch", leave=False)
    steps = 0
    for x, y in progress:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        steps += 1
    return total_loss / max(1, steps)

def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, save_path)
    print(f"Model saved to {save_path}")

def train_loop(model,
    dataloader,
    optimizer,
    criterion,
    device,
    scheduler,
    epochs,
    local_rank,
    n_layers=None,
    d_model=None,
    d_ffn=None,
    save_dir="output",
    mode="pre",        # "pre" or "post"
):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1
    best_model_path = None

    for epoch in range(epochs):
        if dist.is_available() and dist.is_initialized():
            dataloader.sampler.set_epoch(epoch)
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler)
        if local_rank == 0:
            print(f"Loss: {loss:.4f} | Epoch: {epoch+1}")
            if mode == "pre":
                if loss < best_loss and epoch > 6:
                    best_loss = loss
                    best_epoch = epoch + 1
                    filename = (
                        f"E{best_epoch}-{epochs}_N{n_layers}_{d_model}_{d_ffn}_{loss:.4f}.pt"
                    )
                    save_path = os.path.join(save_dir, filename)
                    save_model(model, save_path)
                    print(
                        f"New best model found at epoch {best_epoch}, loss={loss:.4f}. Saved to {save_path}"
                    )
                    best_model_path = save_path
            elif mode == "post":
                filename = (
                    f"E{epoch+1}-{epochs}_N{n_layers}_{d_model}_{d_ffn}_{loss:.4f}_post.pt"
                )
                save_path = os.path.join(save_dir, filename)
                save_model(model, save_path)
    return model

def pre_train(args):
    local_rank, device = get_device()
    max_tokens = args.n_tokens * 1_000_000 if args.n_tokens is not None else None
    dataset, vocab_size = get_dataset(
        split="train", data_dir=args.data_dir, max_tokens=max_tokens, chunk_size=args.chunk_size, is_post=False,
    )
    dataloader = get_data_loader(dataset, args.batch_size)
    nhead = pick_nhead(args.d_model, args.nhead)

    model = get_model(args.arch, vocab_size, args.d_model, args.d_ffn, args.n_layers, args.chunk_size, args.dropout, nhead, device, use_ddp=True, local_rank=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.03 * total_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    criterion = nn.CrossEntropyLoss()

    model = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        local_rank=local_rank,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        save_dir="output",
        mode="pre",
    )

def post_train(args):
    local_rank, device = get_device()
    max_tokens = args.n_tokens * 1_000_000 if args.n_tokens is not None else None
    dataset, vocab_size = get_dataset(
        split="train", data_dir=args.data_dir, max_tokens=max_tokens, chunk_size=args.chunk_size, is_post=True,
    )
    dataloader = get_data_loader(dataset, args.batch_size)
    config = parse_filename(args.model_path) or {
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "d_ffn": args.d_ffn,
        "is_post": is_post
    }
    nhead = pick_nhead(config["d_model"], args.nhead)

    model = get_model(args.arch, vocab_size, config["d_model"], config["d_ffn"], config["n_layers"], args.chunk_size, args.dropout, nhead, device, use_ddp=True, local_rank=local_rank)
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.03 * total_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    criterion = nn.CrossEntropyLoss()

    model = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        local_rank=local_rank,
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        d_ffn=config["d_ffn"],
        save_dir="output",
        mode="post",
    )

def main(args):
    if args.mode == "pre":
        pre_train(args)
    elif args.mode == "post":
        post_train(args)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="decoder", choices=["ffn", "decoder"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="pre", choices=["pre", "post"])
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--n_tokens", type=int, default=None, help="Number of millions of tokens to use from start of dataset. If not set, use full dataset.")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
