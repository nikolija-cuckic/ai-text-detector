"""
Token embedding + Sinusoidal Positional Encoding for the encoder classifier.
Fixed sinusoidal PE from "Attention Is All You Need" (Vaswani et al. 2017), 
which requires no additional parameters and generalizes well.

BertTokenizer already adds [CLS] (token id=101) at position 0 of every sequence.
Treating that existing token as our classification token.
After N encoder blocks, hidden[:, 0, :] is the whole-sequence representation.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from Vaswani et al. 2017.

    For position pos and dimension i:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Stored as a buffer, not a trainable parameter, saved in state_dict.
    Shape: [1, max_len, d_model], broadcastable over batch dimension.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                        # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                          # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims -> cos

        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model] — input + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]  # PE broadcasts over batch
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Full embedding layer: token embedding scaled by sqrt(d_model) + sinusoidal PE.

    Scaling by sqrt(d_model) is from Vaswani et al. — keeps token embedding
    magnitude comparable to the positional encoding values.

    padding_idx=0 means the [PAD] token (id=0) always produces a zero embedding,
    and its gradients are not updated during backprop
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq_len] — from BertTokenizer, [CLS]=101 at position 0
        Returns:
            [B, seq_len, d_model]
        """
        x = self.token_emb(input_ids) * self.scale  # [B, seq_len, d_model]
        return self.pos_enc(x)                       # [B, seq_len, d_model]


# quick sanity check

if __name__ == "__main__":
    torch.manual_seed(9)

    VOCAB_SIZE = 30522  # BertTokenizer vocab size
    D_MODEL = 128
    MAX_LEN = 256
    B = 2
    SEQ_LEN = 32

    emb = TransformerEmbedding(VOCAB_SIZE, D_MODEL, MAX_LEN)

    input_ids = torch.randint(1, VOCAB_SIZE, (B, SEQ_LEN))
    input_ids[:, 0] = 101  # [CLS] at position 0, as BertTokenizer would produce

    out = emb(input_ids)
    print(f"Input  shape: {input_ids.shape}")  # [2, 32]
    print(f"Output shape: {out.shape}")         # [2, 32, 128]
    assert out.shape == (B, SEQ_LEN, D_MODEL)
    print("OK")
