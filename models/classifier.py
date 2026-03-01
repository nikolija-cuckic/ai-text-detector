"""
Transformer encoder classifier built from scratch.
Stacks N EncoderBlocks on top of TransformerEmbedding,
then uses the [CLS] token representation for binary classification.

Architecture:
    input_ids [B, seq_len]
        -> TransformerEmbedding       [B, seq_len, d_model]
        -> EncoderBlock x N           [B, seq_len, d_model]
        -> CLS token [:, 0, :]        [B, d_model]
        -> Dropout -> Linear          [B, 2]
"""

import sys
sys.path.append(".")

import torch
import torch.nn as nn
from layers.embeddings import TransformerEmbedding
from layers.encoder_block import EncoderBlock


class TransformerClassifier(nn.Module):
    """
    From-scratch bidirectional transformer encoder for binary text classification.

    Takes tokenized input from BertTokenizer (so vocab_size=30522 and [CLS] is already at position 0) 
    and produces logits for [human, AI] classes.

    Stores attention weights from all layers for visualization.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            dropout=config.dropout
        )

        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])

        # final LayerNorm after all blocks (standard in BERT)
        self.ln_final = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes, bias=config.bias)

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes linear and embedding weights with small normal distribution.
        Biases are zeroed.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple:
        """
        Args:
            input_ids:      [B, seq_len] — token indices, [CLS]=101 at position 0
            attention_mask: [B, seq_len] — 1 for real tokens, 0 for [PAD]

        Returns:
            logits:       [B, num_classes] — raw scores for each class (not softmaxed)
            all_attentions: list of [B, n_heads, seq_len, seq_len], one per layer
        """
        # embed tokens + positions
        x = self.embedding(input_ids)  # [B, seq_len, d_model]

        # pass through N encoder blocks, collect attention weights
        all_attentions = []
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask)
            all_attentions.append(attn_weights)

        x = self.ln_final(x)

        # take [CLS] token (position 0) as the sequence representation
        cls_repr = x[:, 0, :]          # [B, d_model]
        cls_repr = self.dropout(cls_repr)

        logits = self.classifier(cls_repr)  # [B, num_classes]

        return logits, all_attentions

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# quick sanity check

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Config:
        vocab_size: int = 30522  # BertTokenizer vocab size
        d_model: int = 128
        n_heads: int = 8
        n_layers: int = 4
        max_len: int = 256
        dropout: float = 0.1
        bias: bool = False
        num_classes: int = 2

    torch.manual_seed(9)

    config = Config()
    model = TransformerClassifier(config)

    print(f"Parameters: {model.count_parameters():,}")

    B, T = 2, 32
    input_ids = torch.randint(1, config.vocab_size, (B, T))
    input_ids[:, 0] = 101  # [CLS]
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[:, -5:] = 0  # last 5 are [PAD]

    logits, attentions = model(input_ids, attention_mask)

    print(f"Logits shape     : {logits.shape}")           # [2, 2]
    print(f"Num attn layers  : {len(attentions)}")        # 4
    print(f"Attn[0] shape    : {attentions[0].shape}")    # [2, 8, 32, 32]
    assert logits.shape == (B, config.num_classes)
    print("OK")
