"""
Works with all three models (LSTMClassifier, TransformerClassifier, BERTClassifier)
since they all share the same forward interface: (input_ids, attention_mask) -> (logits, _).
"""

import sys
sys.path.append(".")

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any



class Trainer:
    """
    Trains a binary classifier for a fixed number of epochs with early stopping.

    Saves the best checkpoint (lowest val loss) to checkpoints/{model_name}/best.pt.
    Stops early if val loss does not improve for `patience` consecutive epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # cosine LR schedule — decays lr from config.lr to 0 over max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs
        )

        # gradient clipping — prevents exploding gradients, especially in LSTM
        self.grad_clip = 1.0

        # early stopping state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # checkpoint folder
        self.ckpt_dir = os.path.join("checkpoints", model_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        torch.manual_seed(config.seed)

    def _train_epoch(self) -> tuple:
        """Runs one full pass over the training set. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits, _ = self.model(input_ids, attention_mask)  # [B, 2]
            loss = self.criterion(logits, labels)
            loss.backward()

            # clip gradients before optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        """Runs one full pass over the validation set. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Saves model weights, optimizer state and training metadata."""
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        path = os.path.join(self.ckpt_dir, "best.pt")
        torch.save(ckpt, path)
        print(f"  Checkpoint saved -> {path}")

    def load_best_checkpoint(self) -> None:
        """Loads the best saved checkpoint back into the model."""
        path = os.path.join(self.ckpt_dir, "best.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Best checkpoint loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    def train(self) -> dict:
        """
        Main training loop.

        Runs for at most config.max_epochs epochs.
        Stops early if val loss does not improve for config.patience epochs.
        Loads best checkpoint at the end.

        Returns:
            history dict with lists of train_loss, val_loss, train_acc, val_acc per epoch
        """
        print(f"\nTraining {self.model_name} on {self.device}")
        print(f"  Epochs    : {self.config.max_epochs}")
        print(f"  Patience  : {self.config.patience}")
        print(f"  LR        : {self.config.lr}")
        print(f"  Params    : {self.model.count_parameters():,}\n")

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:02d}/{self.config.max_epochs} "
                f"| train_loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val_loss {val_loss:.4f} acc {val_acc:.4f} "
                f"| {elapsed:.1f}s"
            )

            # check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement ({self.epochs_without_improvement}/{self.config.patience})")

                if self.epochs_without_improvement >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        self.load_best_checkpoint()
        return history
