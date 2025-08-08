import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # LayerNorm before the feedforward block, and then add the residual. Post Norm: return self.ln(x + self.body(x))
        return x + self.body(self.ln(x))


class GPTFFNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_ffn=1024, n_layers=6, max_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_ffn) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.head.weight = self.token_emb.weight # Tying weights


    def forward(self, x):
        B, T = x.size()

        # Token + positional embeddings
        tok_emb = self.token_emb(x)  # (B, T, d_model)
        # pos = torch.arange(T, device=x.device).unsqueeze(0)  # (T,) -> (1, T)
        # pos_emb = self.pos_emb(pos)  # (1, T, d_model)
        # x = tok_emb + pos_emb
        x = tok_emb

        for block in self.blocks:
            x = block(x)

        # Final layer norm and logits
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
