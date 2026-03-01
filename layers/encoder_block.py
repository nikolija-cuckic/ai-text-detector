"""
Single Transformer encoder block: attention -> FFN, both with residual + LayerNorm.

Using pre-norm here, instead of post-norm like in the original paper.

out = x + sublayer(LayerNorm(x))
The original x is added back ("skip connection") so gradients can flow 
directly through the network without vanishing in deep stacks.
"""
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
        x -> LayerNorm -> BidirectionalMHA -> residual
          -> LayerNorm -> FeedForward      -> residual

    Returns both the updated hidden state and attention weights for visualization.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = BidirectionalMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            x:            [B, seq_len, d_model]
            padding_mask: [B, seq_len] — 1 for real tokens, 0 for [PAD]

        Returns:
            x:            [B, seq_len, d_model] — updated representations
            attn_weights: [B, n_heads, seq_len, seq_len]
        """
        # attention sub-layer with pre-norm and residual
        attn_out, attn_weights = self.attn(self.ln1(x), padding_mask)
        x = x + attn_out

        # FFN sub-layer with pre-norm and residual
        x = x + self.ffn(self.ln2(x))

        return x, attn_weights


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
    block = EncoderBlock(config)

    B, T = 2, 32
    x = torch.randn(B, T, config.d_model)

    padding_mask = torch.ones(B, T, dtype=torch.long)
    padding_mask[:, -5:] = 0  # last 5 positions are [PAD]

    out, weights = block(x, padding_mask)
    print(f"Input  shape: {x.shape}")      # [2, 32, 128]
    print(f"Output shape: {out.shape}")     # [2, 32, 128]
    print(f"Attn weights: {weights.shape}") # [2, 8, 32, 32]
    assert out.shape == x.shape
    print("OK")
