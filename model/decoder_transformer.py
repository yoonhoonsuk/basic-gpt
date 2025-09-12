import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, dropout=0.1):
        super().__init__()
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-Norm Self-Attention
        norm_x = self.attn_ln(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        # Pre-Norm Feed-Forward
        norm_x = self.ffn_ln(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_ffn=1024, n_layers=6, max_len=256, dropout=0.1, nhead=8):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(d_model, d_ffn, nhead, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x, return_hidden=False):
        B, T = x.size()

        # Token + positional embeddings
        tok_emb = self.token_emb(x)                         # (B, T, d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0) # (1, T)
        pos_emb = self.pos_emb(pos)                         # (1, T, d_model)
        x = tok_emb + pos_emb

        x = self.drop(x)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        if return_hidden:
            return logits, x  # (B, T, vocab_size), (B, T, d_model)
            
        return logits
