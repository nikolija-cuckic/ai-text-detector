"""
Position-wise Feed-Forward Network (FFN) used inside each encoder block.

Each token's representation is transformed independently through two linear layers 
with a GELU activation in between. 

BERT and most modern transformers use GELU (Gaussian Error Linear Unit).
Unlike ReLU which hard-zeros negative values, GELU smoothly gates them,
works better with the residual connections in deep transformers.

Expansion factor of 4 is from Vaswani et al. 
the hidden layer is 4x wider than d_model, 
giving the network capacity for complex token-wise transformations,
then projects back down to d_model to match the residual stream.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Two-layer FFN applied to each token position independently:
        Linear(d_model -> 4*d_model) -> GELU -> Dropout -> Linear(4*d_model -> d_model) -> Dropout
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.d_model, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model] — same shape, richer token representations
        """
        return self.net(x)


# quick sanity check

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Config:
        d_model: int = 128
        dropout: float = 0.1
        bias: bool = False

    torch.manual_seed(9)

    config = Config()
    ffn = FeedForward(config)

    B, T = 2, 32
    x = torch.randn(B, T, config.d_model)
    out = ffn(x)

    print(f"Input  shape: {x.shape}")   # [2, 32, 128]
    print(f"Output shape: {out.shape}") # [2, 32, 128]
    assert out.shape == x.shape
    print("OK")
