import torch
import torch.nn as nn

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
        return self.ln(x + self.body(x))

class GPTFFNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_ffn=1024, n_layers=6, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # self.use_pos_emb = use_pos_emb
        # if use_pos_emb:
        #     self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_ffn) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        # self.max_len = max_len

    def forward(self, x):
        tok_emb = self.token_emb(x)
        # if self.use_pos_emb:
        #     positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        #     pos_emb = self.pos_emb(positions)
        #     x = tok_emb + pos_emb
        # else:
        x = tok_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
