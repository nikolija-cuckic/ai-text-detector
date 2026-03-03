import sys
sys.path.append(".")

import torch
import torch.nn as nn
from typing import Optional
from layers.attention import BidirectionalMultiHeadAttention
from layers.feedforward import FeedForward


class EncoderBlock(nn.Module):
    """
    One encoder block:
        x -> LayerNorm -> BidirectionalMHA -> residual -> LayerNorm -> FeedForward -> residual
    Returns the updated hidden state and attention weights
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = BidirectionalMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x:            [B, seq_len, d_model]
            padding_mask: [B, seq_len] - 1 for real tokens, 0 for [PAD]

        Returns:
            x:            [B, seq_len, d_model] - updated representations
            attn_weights: [B, n_heads, seq_len, seq_len]
        """
        # attention sub-layer with pre-norm and residual
        attn_out, attn_weights = self.attn(self.ln1(x), padding_mask)
        x = x + attn_out

        # FFN sub-layer with pre-norm and residual
        x = x + self.ffn(self.ln2(x))

        return x, attn_weights
