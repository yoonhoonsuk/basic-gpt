import argparse, os, torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from model.decoder_transformer import GPT
from data.kr3_hf.load import load_kr3_probe_set
from preprocess.tokenizer_bbpe import tokenizer_bbpe, get_tokenizer, get_input_ids
from util.file_utils import parse_filename
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return local_rank, device

def prepare_input_ids(texts, chunk_size=256):
    chunks, vocab_size = tokenizer_bbpe(texts, chunk_size=chunk_size)
    if len(chunks) < len(texts):
        print(f"[WARN] tokenizer_bbpe returned {len(chunks)} < {len(texts)}. Falling back to manual padding.")
        tokenizer = get_tokenizer()
        all_ids = get_input_ids(tokenizer, texts)
        padded = [
            ids[:chunk_size] + [0] * max(0, chunk_size - len(ids))
            for ids in all_ids
        ]
        input_ids = torch.tensor(padded, dtype=torch.long)
    else:
        input_ids = torch.tensor(chunks[:len(texts)], dtype=torch.long)
    return input_ids, vocab_size

def load_gpt_model(model_path, vocab_size, d_model, d_ffn, n_layers, chunk_size, dropout, nhead, device):
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ffn=d_ffn,
        n_layers=n_layers,
        max_len=chunk_size,
        dropout=dropout,
        nhead=nhead,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def extract_hidden_features(model, dataloader, device):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, hidden = model(x, return_hidden=True)
            features.append(hidden[:, -1, :].cpu())
            labels.append(y)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def all_gather_variable_tensor(tensor, device):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_size = torch.tensor([tensor.size(0)], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)
    pad_shape = list(tensor.shape)
    pad_shape[0] = max_size
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=device)
    pad[:tensor.size(0)] = tensor
    gathered = [torch.zeros_like(pad) for _ in range(world_size)]
    dist.all_gather(gathered, pad)
    if rank == 0:
        return torch.cat([g[:s] for g, s in zip(gathered, sizes)], dim=0)
    return None

def get_outfile_name(model_path, outdir="output", ext="_sentiment.pt"):
    model_base = os.path.basename(model_path)
    stem, _ = os.path.splitext(model_base)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"{stem}{ext}")

def main(args):
    local_rank, device = setup_ddp()
    probe_dataset = load_kr3_probe_set(limit=args.limit)
    texts = list(probe_dataset['text'])
    labels = torch.tensor(list(probe_dataset['label']), dtype=torch.long)
    input_ids, vocab_size = prepare_input_ids(texts, chunk_size=args.chunk_size)
    dataset = TensorDataset(input_ids, labels)
    sampler = DistributedSampler(dataset) if dist.is_available() and dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None))

    config = parse_filename(args.model_path) or {
        "n_layers": args.n_layers, "d_model": args.d_model, "d_ffn": args.d_ffn, "is_post": False
    }
    model = load_gpt_model(
        args.model_path, vocab_size,
        config["d_model"], config["d_ffn"], config["n_layers"],
        args.chunk_size, args.dropout, args.nhead, device
    )
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    features, labels_out = extract_hidden_features(model, dataloader, device)

    if dist.is_available() and dist.is_initialized():
        features_gathered = all_gather_variable_tensor(features, device)
        labels_gathered = all_gather_variable_tensor(labels_out, device)
        dist.barrier()
        is_main = dist.get_rank() == 0
    else:
        features_gathered, labels_gathered = features, labels_out
        is_main = True

    if is_main:
        X = features_gathered.cpu().numpy()
        y = labels_gathered.cpu().numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Validation accuracy:", acc)
        print(classification_report(y_test, y_pred))
        out_probe = get_outfile_name(args.model_path, args.output, ext="_probe.pkl")
        joblib.dump({'probe': clf, 'scaler': scaler}, out_probe)
        print(f"[INFO] Saved linear probe (with scaler) to {out_probe}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="output", help="Output directory (not file name!)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of probe samples (for quick runs)")
    args = parser.parse_args()
    main(args)
