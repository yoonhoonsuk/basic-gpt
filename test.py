import argparse, os, re, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset as hf_load

from preprocess.manual_bbpe import manual_bbpe
from preprocess.tokenizer_bbpe import tokenizer_bbpe
from model.FFNN import GPTFFNN
from model.decoder_transformer import GPT

from util.file_utils import parse_filename

def load_dataset(name):
    if name == "movie_review_ko":
        ds = hf_load(
            "csv",
            data_files=[
                "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
            ],
            delimiter="\t",
        )["train"]
        return {"texts": [t for t in ds["document"] if isinstance(t, str) and t.strip()]}

    if name == "hate_speech_ko":
        from datasets import concatenate_datasets
        dd = hf_load("nayohan/korean-hate-speech")
        ds = concatenate_datasets(list(dd.values())) if hasattr(dd, "values") else dd
        col = next((c for c in ("comments","comment","text","document","sentence","content") if c in ds.column_names), ds.column_names[0])
        return {"texts": [t for t in ds[col] if isinstance(t, str) and t.strip()]}

    raise ValueError("dataset must be 'movie_review_ko' or 'hate_speech_ko'")

def get_token(texts, mode="tokenizer", chunk_size=256):
    return manual_bbpe(texts, chunk_size) if mode == "manual" else tokenizer_bbpe(texts, chunk_size)

def get_data_loader(chunks, batch_size):
    x = torch.tensor([c[:-1] for c in chunks], dtype=torch.long)
    y = torch.tensor([c[1:] for c in chunks], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()

def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = ["movie_review_ko","hate_speech_ko"]
    for ds_name in names:
        dataset = load_dataset(ds_name)
        texts = dataset["texts"]
        chunks, vocab_size = get_token(texts, mode=args.mode, chunk_size=args.chunk_size)
        dataloader = get_data_loader(chunks, args.batch_size)

        state = torch.load(args.model_path, map_location=device)
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        if args.arch == "decoder":
            ModelCls = GPT
        elif args.arch == "ffn":
            ModelCls = GPTFFNN
        else:
            is_decoder = isinstance(state, dict) and any(("attn." in k or "pos_emb." in k) for k in state.keys())
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
        avg_loss, ppl = evaluate(model, dataloader, criterion, device)
        print(f"{ds_name} Loss: {avg_loss:.4f}, Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--mode", type=str, default="tokenizer", choices=["tokenizer", "manual"])
    parser.add_argument("--model_path", type=str, default="output/gpt_model_ddp.pt")
    parser.add_argument("--arch", type=str, choices=["ffn", "decoder"], default=None, 
                        help="Force model type; if omitted, auto-detect from checkpoint.")
    args = parser.parse_args()

    parsed = parse_filename(args.model_path)
    if parsed:
        args.n_layers = parsed["n_layers"]
        args.d_model  = parsed["d_model"]
        args.d_ffn    = parsed["d_ffn"]
        print(f"[validate] Parsed from filename: "
              f"N{args.n_layers} d_model={args.d_model} d_ffn={args.d_ffn}")

    main(args)
