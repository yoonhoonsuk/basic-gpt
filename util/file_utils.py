import os
import re

def parse_filename(path: str):
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    # Pattern: E{best_epoch}-{epochs}_N{n_layers}_{d_model}_{d_ffn}_{loss}[optional:_post]
    m = re.match(
        r"^E(?P<best_epoch>\d+)-(?P<epochs>\d+)_N(?P<n_layers>\d+)_(?P<d_model>\d+)_(?P<d_ffn>\d+)_?(?P<loss>[\d\.]+)?(?P<post>_post)?$",
        stem
    )
    if not m:
        return None
    g = m.groupdict()
    return {
        "n_layers": int(g["n_layers"]),
        "d_model": int(g["d_model"]),
        "d_ffn": int(g["d_ffn"]),
        "is_post": bool(g["post"])
    }
