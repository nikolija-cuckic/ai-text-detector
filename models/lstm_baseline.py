import sys
sys.path.append(".")

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.
    Uses the final hidden states from both directions as the sequence representation
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.lstm_embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=config.lstm_embed_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,       # input/output shape is [B, T, features], not [T, B, features]
            bidirectional=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0
        )
        # output dim is 2*hidden_dim because we concat forward and backward hidden states
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(2 * config.lstm_hidden_dim, config.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)  # helps with vanishing gradients
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len] used to find the last real token per sequence
        Returns:
            logits: [B, num_classes]
        """
        x = self.embedding(input_ids) # [B, seq_len, embed_dim]

        # pack padded sequences so LSTM ignores [PAD] tokens
        lengths = attention_mask.sum(dim=1).cpu()  # actual length of each sequence in batch
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        packed_out, (hidden, _) = self.lstm(packed)

        # hidden shape: [num_layers * num_directions, B, hidden_dim]
        # forward:  hidden[-2]  (last layer, forward direction)
        # backward: hidden[-1]  (last layer, backward direction)
        fwd = hidden[-2]  # [B, hidden_dim]
        bwd = hidden[-1]  # [B, hidden_dim]

        combined = torch.cat([fwd, bwd], dim=1)  # [B, 2*hidden_dim]
        combined = self.dropout(combined)

        logits = self.classifier(combined)  # [B, num_classes]

        return logits, None  # None keeps same interface as TransformerClassifier

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)