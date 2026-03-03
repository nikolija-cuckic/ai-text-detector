import sys
sys.path.append(".")

import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    """
    Pretrained BERT encoder + custom classification head.
    Only the head is trained from scratch, everything else starts from pretrained values
    """

    def __init__(self, config) -> None:
        super().__init__()

        # pretrained encoder - loads weights from HuggingFace cache
        self.bert = BertModel.from_pretrained(
            config.bert_model_name,
            attn_implementation="eager"  # needed for output_attentions=True
        )

        bert_hidden_dim = self.bert.config.hidden_size  # 768 for bert-base-uncased

        # custom classification head 
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(bert_hidden_dim, config.num_classes)

        # only initialize the head
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len] passed to BERT so it ignores [PAD]

        Returns:
            logits: [B, num_classes]
            attentions: tuple of [B, n_heads, seq_len, seq_len] per layer
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        # outputs.last_hidden_state: [B, seq_len, 768]
        # outputs.attentions: tuple of 12 tensors, each [B, 12, seq_len, seq_len]
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [B, 768] [CLS] token
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr) # [B, num_classes]

        return logits, outputs.attentions

    def freeze_bert(self) -> None:
        """Freezes all BERT parameters so only the classification head is trained."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self) -> None:
        """Unfreezes all BERT parameters for full fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())