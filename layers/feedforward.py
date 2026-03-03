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
            [B, seq_len, d_model] same shape but better token representations
        """
        return self.net(x)
