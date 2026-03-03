import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from Vaswani et al 2017
    For position pos and dimension i:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Stored as a buffer, not trainable parameter, saved in state_dict
    Shape: [1, max_len, d_model], broadcastable over batch dim
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))# [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)# even -> sin
        pe[:, 1::2] = torch.cos(position * div_term)# odd -> cos

        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model] input + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]  # PE broadcasts over batch
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Full embedding layer: token embedding scaled by sqrt(d_model) + sinusoidal PE.
    padding_idx=0 means the [PAD] token (id=0) always gives a zero embedding,
    and its gradients wont be updated during backprop
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq_len] from BertTokenizer, [CLS]=101 at position 0
        Returns:
            [B, seq_len, d_model]
        """
        x = self.token_emb(input_ids) * self.scale  # [B, seq_len, d_model]
        return self.pos_enc(x) # [B, seq_len, d_model]
