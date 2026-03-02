import sys
sys.path.append(".")

import argparse
import json
import os
import torch

from configs.config import DataConfig, LSTMConfig, TransformerConfig, BERTConfig
from data.dataset import build_dataloaders
from models.lstm_baseline import LSTMClassifier
from models.classifier import TransformerClassifier
from models.bert_classifier import BERTClassifier
from training.trainer import Trainer

# use all physical cores on CPU
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_history(history: dict, model_name: str) -> None:
    """Saves training history to results/{model_name}_history.json for later evaluation."""
    path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved -> {path}")


def train_lstm(data_cfg, lstm_cfg, train_loader, val_loader) -> dict:
    model = LSTMClassifier(config=lstm_cfg)  # samo config, bez kwargs
    model_name = f"lstm_maxlen{data_cfg.max_len}"
    trainer = Trainer(
        model=model,
        config=lstm_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
    )
    history = trainer.train()
    save_history(history, model_name)
    return history


def train_transformer(data_cfg, trans_cfg, train_loader, val_loader) -> dict:
    model = TransformerClassifier(config=trans_cfg)  # samo config
    model_name = f"transformer_maxlen{data_cfg.max_len}"
    trainer = Trainer(
        model=model,
        config=trans_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
    )
    history = trainer.train()
    save_history(history, model_name)
    return history


def train_bert(data_cfg, bert_cfg, train_loader, val_loader, frozen: bool = False) -> dict:
    model = BERTClassifier(config=bert_cfg)
    if frozen:
        model.freeze_bert()
    suffix = "frozen" if frozen else "full"
    model_name = f"bert_{suffix}_maxlen{data_cfg.max_len}"
    trainer = Trainer(
        model=model,
        config=bert_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
    )
    history = trainer.train()
    save_history(history, model_name)
    return history



def parse_args() -> argparse.Namespace:
    """
    CLI overrides for configs defined in main().

    Example usage:
        python train.py --models lstm transformer
        python train.py --models bert --max_epochs 3 --batch_size 16
        python train.py --models lstm --lr 5e-4 --patience 5 --max_len 128
    """
    parser = argparse.ArgumentParser(description="Train AI text detection models")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "transformer", "bert"],
        default=None,
        help="Which models to train. Defaults to those enabled in main().",
    )

    # data overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)

    # training overrides (applied to all selected models)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    return parser.parse_args()


def apply_overrides(args: argparse.Namespace, data_cfg, lstm_cfg, trans_cfg, bert_cfg) -> None:
    """Applies CLI overrides to all relevant config objects in-place."""
    training_cfgs = [lstm_cfg, trans_cfg, bert_cfg]

    if args.batch_size is not None:
        data_cfg.batch_size = args.batch_size

    if args.max_len is not None:
        data_cfg.max_len = args.max_len
        trans_cfg.max_len = args.max_len  # TransformerClassifier needs max_len too

    for cfg in training_cfgs:
        if args.max_epochs is not None:
            cfg.max_epochs = args.max_epochs
        if args.patience is not None:
            cfg.patience = args.patience
        if args.lr is not None:
            cfg.lr = args.lr


def main() -> None:
    args = parse_args()

    # base configs — edit these to change defaults without CLI
    data_cfg = DataConfig()
    lstm_cfg = LSTMConfig()
    trans_cfg = TransformerConfig()
    bert_cfg = BERTConfig()

    apply_overrides(args, data_cfg, lstm_cfg, trans_cfg, bert_cfg)

    # which models to train — CLI overrides this list if --models is passed
    # to train only specific models, edit this list or use --models from CLI
    models_to_train = args.models if args.models is not None else [ "transformer"] #"lstm", , "bert"

    # build dataloaders once — all models share the same data pipeline
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        batch_size=data_cfg.batch_size,
        max_len=data_cfg.max_len,
        val_split=data_cfg.val_split,
        test_split=data_cfg.test_split,
        num_workers=data_cfg.num_workers,
        seed=data_cfg.seed,
        data_path=data_cfg.data_path,
    )

    # save test_loader for evaluate.py
    test_loader_path = os.path.join(RESULTS_DIR, "test_loader.pt")
    torch.save(test_loader, test_loader_path)
    print(f"Test loader saved -> {test_loader_path}\n")

    if "lstm" in models_to_train:
        train_lstm(data_cfg, lstm_cfg, train_loader, val_loader)

    if "transformer" in models_to_train:
        train_transformer(data_cfg, trans_cfg, train_loader, val_loader)

    if "bert" in models_to_train:
        train_bert(data_cfg, bert_cfg, train_loader, val_loader, frozen=True)
        train_bert(data_cfg, bert_cfg, train_loader, val_loader, frozen=False)

    print("\nAll done. Run evaluate.py to see test results.")


if __name__ == "__main__":
    main()
