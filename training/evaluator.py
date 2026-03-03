import sys
sys.path.append(".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Any


class Evaluator:
    """Works with all three models (LSTM, TransformerClassifier, BERTClassifier)
    because they all return (logits, _)"""

    def __init__(self, model: nn.Module, device: str) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple:
        """
        Runs inference on the full loader and returns predictions and true labels.
        Returns:
            preds:  list of predicted class indices (0=human, 1=AI)
            probs:  list of predicted probabilities for class 1 (AI) for ROC-AUC
            labels: list of true class indices
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(input_ids, attention_mask)  # [B, 2]

            probs = torch.softmax(logits, dim=-1)[:, 1] # P(AI) - [B]
            preds = logits.argmax(dim=-1) # [B]

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        return all_preds, all_probs, all_labels

    def evaluate(self, loader: DataLoader, split_name: str = "test") -> dict:
        """
        Args:
            loader: DataLoader to evaluate on
            split_name: label for printing ("val" or "test")
        Returns:
            dict with accuracy, f1, precision, recall, roc_auc, confusion_matrix
        """
        preds, probs, labels = self.predict(loader)
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
            "precision": precision_score(labels, preds, average="binary"),
            "recall": recall_score(labels, preds, average="binary"),
            "roc_auc": roc_auc_score(labels, probs),
            "confusion_matrix": confusion_matrix(labels, preds)
        }
        self._print_report(metrics, labels, preds, split_name)
        return metrics

    def _print_report(self, metrics: dict, labels: list, preds: list, split_name: str) -> None:
        print(f"\nEvaluation on {split_name} set:")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"\nConfusion matrix (rows=true, cols=pred):")
        cm = metrics["confusion_matrix"]
        print(f"           Human  AI")
        print(f"  Human  [ {cm[0][0]:5d}  {cm[0][1]:5d} ]")
        print(f"  AI     [ {cm[1][0]:5d}  {cm[1][1]:5d} ]")
        print(f"\nClassification report:")
        print(classification_report(labels, preds, target_names=["human", "AI"]))
