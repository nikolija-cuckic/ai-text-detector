"""
Bidirectional multi-head self-attention for text classification.

NO causal mask — every token sees all others in both directions. 
Only mask applied is the padding mask, which prevents attention to [PAD] tokens.

separate q_proj, k_proj, v_proj (MultiHeadAttention style),
and return attention weights they can be visualize later.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class BidirectionalMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention without causal mask (bidirectional).
    [PAD] tokens are masked out via the padding mask from the dataloader.

    Returns both output and attention weights (needed for visualization).
    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads: int = config.n_heads
        self.d_model: int = config.d_model
        self.head_dim: int = config.d_model // config.n_heads
        self.dropout: float = config.dropout

        # separate projections for Q, K, V 
        self.q_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.out_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.resid_dropout: nn.Dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            x:            [B, seq_len, d_model] — input from embedding or previous block
            padding_mask: [B, seq_len] — 1 for real tokens, 0 for [PAD] (from dataloader)

        Returns:
            out:          [B, seq_len, d_model] — attended representations
            attn_weights: [B, n_heads, seq_len, seq_len] — attention distributions (for viz)
        """
        B: int
        T: int
        C: int
        B, T, C = x.size()

        # compute Q, K, V
        q: torch.Tensor = self.q_proj(x)  # [B, T, d_model]
        k: torch.Tensor = self.k_proj(x)  # [B, T, d_model]
        v: torch.Tensor = self.v_proj(x)  # [B, T, d_model]

        # split into heads: (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention scores
        # (B, n_heads, T, head_dim) x (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        scores: torch.Tensor = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # apply padding mask - mask out [PAD] positions so they get zero attention weight
        # padding_mask is [B, T]: expand to [B, 1, 1, T] for broadcasting over heads and queries
        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2) == 0,
                float("-inf")
            )

        # softmax over last dim (key dimension) — each query gets a distribution over keys
        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)  # [B, n_heads, T, T]

        # if a row is all -inf (full padding), softmax gives NaN — replace with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.attn_dropout(attn_weights)

        # weighted sum of values
        out: torch.Tensor = attn_weights @ v  # [B, n_heads, T, head_dim]

        # merge heads: (B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, attn_weights


# quick sanity check

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from dataclasses import dataclass

    @dataclass
    class Config:
        d_model: int = 128
        n_heads: int = 8
        dropout: float = 0.1
        bias: bool = False

    torch.manual_seed(9)

    config = Config()
    attn = BidirectionalMultiHeadAttention(config)

    B, T = 2, 32
    x = torch.randn(B, T, config.d_model)

    # simulate padding mask: last 5 tokens are [PAD]
    padding_mask = torch.ones(B, T, dtype=torch.long)
    padding_mask[:, -5:] = 0

    out, weights = attn(x, padding_mask)
    print(f"Input  shape: {x.shape}")           # [2, 32, 128]
    print(f"Output shape: {out.shape}")          # [2, 32, 128]
    print(f"Attn weights: {weights.shape}")      # [2, 8, 32, 32]

    # verify padding positions have zero attention weight
    pad_attn = weights[:, :, :, -5:]
    print(f"Attention on PAD tokens (should be 0): {pad_attn.abs().max().item():.6f}")

    assert out.shape == (B, T, config.d_model)
    assert weights.shape == (B, config.n_heads, T, T)
    print("OK")
