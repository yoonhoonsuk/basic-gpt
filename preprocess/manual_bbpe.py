import unicodedata
import regex as re
import torch
import os
import json
from typing import Dict, Tuple


def get_stats(ids: torch.Tensor, V: int) -> torch.Tensor:
    """
    Count bigrams (compute a unique index) in the sequence using hash: a * V + b
    Returns a 1D tensor of size V*V with counts (torch.bincount(pair_keys, minlength=V*V))
    Returns a tensor of size V*V where index i = a*V + b maps back to (a, b)
    """
    if ids.numel() < 2:
        return torch.zeros(1, dtype=torch.int32, device=ids.device)
    pair_keys = ids[:-1] * V + ids[1:]
    return torch.bincount(pair_keys, minlength=V * V)


def merge_vocab(ids: torch.Tensor, pair: Tuple[int, int], new_idx: int) -> torch.Tensor:
    """
    Replace all occurrences of (pair[0], pair[1]) with new_idx.
    mask is a boolean tensor identifying where the target pair occurs.

    ids[:-1][mask] = new_idx: updates the first element of each matched pair to new_idx.
    keep[1:][mask] = False: removes the second element of each matched pair.
    Returns a tensor where every (a, b) is replaced by new_idx
    """
    p0, p1 = pair
    mask = (ids[:-1] == p0) & (ids[1:] == p1)
    if not mask.any():
        return ids

    ids = ids.clone()
    ids[:-1][mask] = new_idx 

    keep = torch.ones_like(ids, dtype=torch.bool)
    keep[1:][mask] = False
    return ids[keep]


def replace_control_characters(s: str) -> str:
    return "".join(
        ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}"
        for ch in s
    )

def render_token(t: bytes) -> str:
    return replace_control_characters(t.decode('utf-8', errors='replace'))


def dump_tokenizer_outputs(
    chunks, tokenizer, token2id, output_dir="output", prefix="run"
):
    os.makedirs(output_dir, exist_ok=True)
    chunk_path = os.path.join(output_dir, f"{prefix}_chunks.txt")
    with open(chunk_path, "w", encoding="utf-8") as f:
        for c in chunks:
            ids = c.tolist() if hasattr(c, "tolist") else list(c)
            f.write(" ".join(map(str, ids)) + "\n")
    print(f"Chunks written to {chunk_path}")

    vocab_path = os.path.join(output_dir, f"{prefix}_vocab.json")
    vocab_dump = {idx: render_token(token) for idx, token in tokenizer.vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dump, f, ensure_ascii=False, indent=2)
    print(f"Vocab written to {vocab_path}")

    merges_path = os.path.join(output_dir, f"{prefix}_merges.json")
    merges_dump = {str(k): v for k, v in tokenizer.merges.items()}
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_dump, f, ensure_ascii=False, indent=2)
    print(f"Merges written to {merges_path}")

    token2id_path = os.path.join(output_dir, f"{prefix}_token2id.json")
    token2id_dump = {render_token(k): v for k, v in token2id.items()}
    with open(token2id_path, "w", encoding="utf-8") as f:
        json.dump(token2id_dump, f, ensure_ascii=False, indent=2)
    print(f"Token2id written to {token2id_path}")

class Tokenizer:
    pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self, vocab_size: int = 1000, device: str = 'cuda'):
        """
        vocab: maps int IDs to byte strings (tokens).
        merges: maps token pairs to merged token ID.
        inverse_special_tokens: reserved for custom tokens (unused in basic BPE).
        pattern: regex pattern for tokenization (not used here, but could be used for pre-tokenization).
        """
        self.vocab_size = vocab_size
        self.device = device
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.inverse_special_tokens: Dict[int, str] = {}

    def train(self, text: str, vocab_size: int = None, verbose: bool = False):
        """
        Build a vocabulary of merged tokens using BPE until reaching the target vocab_size.
        Start with raw UTF-8 bytes (ids of integers 0–255).
        Repeatedly:
            Call get_stats to count bigrams.
            Find the most frequent bigram.
            Merge it into a new token via merge_vocab.
            Update vocab and merges.
        Stops if:
            No more frequent pairs.
            Merged pair already exists.
        """
        vocab_size = vocab_size or self.vocab_size
        ids = torch.tensor(list(text.encode('utf-8')), dtype=torch.int32, device=self.device)
        V = 256

        while V < vocab_size:
            stats = get_stats(ids, V) # maps pair_keys (a * v + b) : counts
            pair_key = int(stats.argmax())
            if stats[pair_key] == 0:
                break

            p0, p1 = divmod(pair_key, V)
            if (p0, p1) in self.merges:
                break  # avoid repeated merges

            ids = merge_vocab(ids, (p0, p1), V)
            self.merges[(p0, p1)] = V
            self.vocab[V] = self.vocab[p0] + self.vocab[p1]
            if verbose:
                print(f"Merged ({p0}, {p1}) -> {V} [{render_token(self.vocab[V])}]")
            V += 1

    def encode(self, text: str) -> torch.Tensor:
        """
        Convert input text to a tensor of token IDs using the trained merge rules.
        Encodes the string into raw bytes (ints 0–255).
        Builds a rank matrix to prioritize merges.
        While mergeable pairs exist:
        Compute pair indices: pair_keys = ids[:-1] * V + ids[1:]
        Get their ranks.
        Merge the lowest rank pair (most important first).
        """
        ids = torch.tensor(list(text.encode()), dtype=torch.int32, device=self.device)
        V = 256 + len(self.merges)

        rank = torch.full((V, V), -1, dtype=torch.int16, device=self.device)
        for r, ((a, b), _) in enumerate(self.merges.items()):
            rank[a, b] = r

        while True:
            if ids.numel() < 2:
                break
            pair_keys = ids[:-1] * V + ids[1:]
            r = rank.view(-1)[pair_keys]
            best = (r >= 0).nonzero(as_tuple=False)
            if best.numel() == 0:
                break

            pos = best[r[best].argmin()][0]  # index of best pair
            a, b = int(ids[pos]), int(ids[pos + 1])
            new_idx = self.merges.get((a, b))
            if new_idx is None:
                break
            ids = merge_vocab(ids, (a, b), new_idx)

        return ids

    def decode(self, ids):
        """
        Convert a list of token IDs back into a string.
        """
        decoded_bytes = []
        for idx in ids:
            idx = int(idx)
            if idx in self.vocab:
                decoded_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                decoded_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        return b"".join(decoded_bytes).decode("utf-8", errors="replace")

    def render_vocab(self):
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        for idx, token in self.vocab.items():
            token_str = render_token(token)
            if idx in inverted_merges:
                i0, i1 = inverted_merges[idx]
                s0, s1 = render_token(self.vocab[i0]), render_token(self.vocab[i1])
                print(f"[{s0}][{s1}] -> [{token_str}] {idx}")
            else:
                print(f"[{token_str}] {idx}")


def manual_bbpe(texts, vocab_size=512, chunk_size=1024, verbose=False, device='cuda'):
    """
    A high-level API to tokenize multiple texts using the Tokenize
    Returns:
        chunks: List of token ID tensors.
        merges: Dict mapping (a, b) → new token index.
        token2id: Dict mapping byte strings → token IDs.
        tokenizer: The trained Tokenizer instance.
    """
    assert vocab_size >= 256
    assert isinstance(texts, list) and all(isinstance(x, str) for x in texts)

    normalized_text = unicodedata.normalize("NFKC", "\n".join(texts))
    tokenizer = Tokenizer(vocab_size=vocab_size, device=device)
    tokenizer.train(normalized_text, verbose=verbose)

    all_ids = tokenizer.encode(normalized_text)
    num_chunks = len(all_ids) // chunk_size
    chunks = [all_ids[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    token2id = {v: k for k, v in tokenizer.vocab.items()}
    print(len(tokenizer.vocab))
    dump_tokenizer_outputs(chunks, tokenizer, token2id, output_dir="output", prefix = "manual_bbpe")
    return chunks, tokenizer.merges, token2id, tokenizer
