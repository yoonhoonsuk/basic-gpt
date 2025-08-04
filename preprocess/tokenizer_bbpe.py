from transformers import AutoTokenizer
from itertools import chain
import unicodedata

def tokenizer_bbpe(texts, chunk_size=256, verbose=False):
    norm_texts = [unicodedata.normalize("NFKC", t) for t in texts]

    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

    input_ids = tokenizer(norm_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]

    all_ids = list(chain(*input_ids))
    num_chunks = len(all_ids) // chunk_size
    chunks = [all_ids[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    if verbose:
        print(f"[INFO] Made {len(chunks)} chunks of {chunk_size} tokens each.")
    # Return both chunks and vocab size
    return chunks, tokenizer.vocab_size
