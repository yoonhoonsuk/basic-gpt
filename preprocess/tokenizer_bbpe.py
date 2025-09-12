from transformers import AutoTokenizer
from itertools import chain
import unicodedata
import os

cache_dir = "./hf_cache"

def get_tokenizer():
    token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if token:
        from huggingface_hub import login
        login(token)
    tokenizer = AutoTokenizer.from_pretrained(
        "skt/kogpt2-base-v2",
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
    return tokenizer

def get_input_ids(tokenizer, texts):
    norm_texts = [unicodedata.normalize("NFKC", t) for t in texts]
    return tokenizer(norm_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]

def tokenizer_bbpe(texts, chunk_size=256, verbose=False):
    tokenizer = get_tokenizer()
    input_ids = get_input_ids(tokenizer, texts)
    all_ids = list(chain(*input_ids))
    if chunk_size is None:
        # No chunking, return one chunk per input
        chunks = [ids for ids in input_ids]
    else:
        num_chunks = len(all_ids) // chunk_size
        chunks = [all_ids[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    if verbose:
        print(f"[INFO] Made {len(chunks)} chunks of {chunk_size} tokens each.")
    return chunks, tokenizer.vocab_size
