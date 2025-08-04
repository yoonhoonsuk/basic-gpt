from collections import defaultdict, Counter
import unicodedata

def get_vocab(texts):
    vocab = defaultdict(int)
    for text in texts:
        norm_text = unicodedata.normalize("NFKC", text)  # Normalize each text
        for word in norm_text.strip().split(): # Pre-tokenize
            tokens = tuple(list(word) + ["</w>"])
            vocab[tokens] += 1
    return vocab

def get_pairs(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in v_in.items():
        word_str = ' '.join(word)
        new_word_str = word_str.replace(bigram, replacement)
        new_word = tuple(new_word_str.split())
        v_out[new_word] = freq
    return v_out

def manual_bpe(texts, vocab_size=50, chunk_size=256, verbose=False):
    # Normalize all input texts before further processing
    texts = [unicodedata.normalize("NFKC", text) for text in texts]

    vocab = get_vocab(texts)
    merges = []
    while True:
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
        if verbose:
            print(f"Merged {best}, number of merges: {len(merges)}")
        if len(set([tok for word in vocab for tok in word])) >= vocab_size:
            break

    def encode_word(word):
        chars = list(word) + ["</w>"]
        for pair in merges:
            i = 0
            while i < len(chars) - 1:
                if (chars[i], chars[i+1]) == pair:
                    chars[i:i+2] = [''.join(pair)]
                else:
                    i += 1
        return chars

    all_tokens = []
    for text in texts:
        text = unicodedata.normalize("NFKC", text)  # Normalize again before encoding
        for word in text.strip().split():
            all_tokens.extend(encode_word(word))

    vocab_set = sorted(set(all_tokens))
    token2id = {tok: i for i, tok in enumerate(vocab_set)}
    all_ids = [token2id[tok] for tok in all_tokens]

    num_chunks = len(all_ids) // chunk_size
    chunks = [all_ids[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    return chunks
