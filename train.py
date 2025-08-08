import os
import argparse
import time
import torch
import torch.nn as nn
from preprocess.manual_bbpe import manual_bbpe
from preprocess.tokenizer_bbpe import tokenizer_bbpe
from data.kr3_hf.load import load_kr3
from model.FFNN import GPTFFNN
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_dataset(split="train", data_path=None):
    if data_path and os.path.exists(data_path):
        return torch.load(data_path)
    else:
        dataset = load_kr3(split=split)
        return {"texts": list(dataset["text"])}

def get_token(texts, mode="tokenizer", chunk_size=256):
    if mode == "manual":
        return manual_bbpe(texts, chunk_size=chunk_size)
    else:
        return tokenizer_bbpe(texts, chunk_size=chunk_size)

def get_data_loader(chunks, batch_size):
    x = torch.tensor([c[:-1] for c in chunks], dtype=torch.long)
    y = torch.tensor([c[1:] for c in chunks], dtype=torch.long)
    data = TensorDataset(x, y)
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(data)
        return DataLoader(data, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(data, batch_size=batch_size, shuffle=True)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    # Distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_dict = get_dataset(data_path=args.data_path)
    texts = dataset_dict["texts"]

    # Tokenize
    chunks, vocab_size = get_token(texts, mode=args.mode, chunk_size=args.chunk_size)
    # total_tokens = sum(len(seq) for seq in chunks)
    # print(f"Total tokens in dataset: {total_tokens:,}")
    # print(f"Suggested parameter budget (Chinchilla rule): ~{suggested_params:,} parameters")
    dataloader = get_data_loader(chunks, args.batch_size)

    # Model
    model = GPTFFNN(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        max_len=args.chunk_size
    ).to(device)

    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,)
    # weight_decay=0.01
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    #label_smoothing=0.1
    
    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        if dist.is_available() and dist.is_initialized():
            dataloader.sampler.set_epoch(epoch)
        loss = train(model, dataloader, optimizer, criterion, device)
        if local_rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        scheduler.step()

    # Save model
    if local_rank == 0:
        print(f"Training took {time.time() - start_time:.2f} seconds.")
        save_path = "output/gpt_model_ddp.pt" if dist.is_initialized() else "output/gpt_model.pt"
        torch.save(model.module.state_dict() if dist.is_initialized() else model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dataset.pt", help="Path to preprocessed dataset (.pt)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=256, help="Sequence length (context window)")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding/hidden size")
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="tokenizer", choices=["tokenizer", "manual"])
    args = parser.parse_args()
    main(args)
