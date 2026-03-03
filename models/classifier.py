import sys
sys.path.append(".")

import torch
import torch.nn as nn
from layers.embeddings import TransformerEmbedding
from layers.encoder_block import EncoderBlock


class TransformerClassifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            dropout=config.dropout
        )

        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])

        # final LayerNorm after all blocks
        self.ln_final = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes, bias=config.bias)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes linear and embedding weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Args:
            input_ids:[B, seq_len] token indices, [CLS]=101 at position 0
            attention_mask: [B, seq_len] 1 for real tokens, 0 for [PAD]

        Returns:
            logits: [B, num_classes] raw scores for each class (not softmaxed)
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
        cls_repr = x[:, 0, :]  # [B, d_model]
        cls_repr = self.dropout(cls_repr)

        logits = self.classifier(cls_repr)  # [B, num_classes]

        return logits, all_attentions

    def count_parameters(self) -> int:
        """total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)