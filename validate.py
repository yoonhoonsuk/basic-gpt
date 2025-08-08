import argparse, os, torch
import torch.nn as nn
from preprocess.manual_bbpe import manual_bbpe
from preprocess.tokenizer_bbpe import tokenizer_bbpe
from model.FFNN import GPTFFNN
from torch.utils.data import DataLoader, TensorDataset

def get_dataset(split="validation", data_dir="data"):
    path = os.path.join(data_dir, f"{split}.pt")
    if os.path.exists(path):
        return torch.load(path)
    raise FileNotFoundError(f"{path} not found. Run load.py first.")

def get_token(texts, mode="tokenizer", chunk_size=256):
    return manual_bbpe(texts, chunk_size) if mode == "manual" else tokenizer_bbpe(texts, chunk_size)

def get_data_loader(chunks, batch_size):
    x = torch.tensor([c[:-1] for c in chunks], dtype=torch.long)
    y = torch.tensor([c[1:] for c in chunks], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataset
    dataset_dict = get_dataset(split="validation", data_dir=args.data_dir)
    texts = dataset_dict["texts"]

    # Tokenize
    chunks, vocab_size = get_token(texts, mode=args.mode, chunk_size=args.chunk_size)
    dataloader = get_data_loader(chunks, args.batch_size)

    # Load model
    model = GPTFFNN(vocab_size=vocab_size, d_model=args.d_model,
                    d_ffn=args.d_ffn, n_layers=args.n_layers,
                    max_len=args.chunk_size).to(device)
    state = torch.load(args.model_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    # Validate
    avg_loss, ppl = validate(model, dataloader, criterion, device)
    print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--mode", type=str, default="tokenizer", choices=["tokenizer", "manual"])
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
