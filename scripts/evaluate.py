import sys
sys.path.append(".")

import argparse
import json
import os
import torch

from configs.config import LSTMConfig, TransformerConfig, BERTConfig
from models.lstm_baseline import LSTMClassifier
from models.classifier import TransformerClassifier
from models.bert_classifier import BERTClassifier
from training.evaluator import Evaluator

RESULTS_DIR = "results"
CKPT_DIR = "checkpoints"

MODEL_MAP = {
    "lstm_maxlen128": (LSTMClassifier, LSTMConfig),
    "lstm_maxlen256": (LSTMClassifier, LSTMConfig),
    "transformer_maxlen128": (TransformerClassifier, TransformerConfig),
    "transformer_maxlen256": (TransformerClassifier, TransformerConfig),
    "bert_maxlen128": (BERTClassifier, BERTConfig),
    "bert_maxlen256": (BERTClassifier, BERTConfig),
}


def load_model(model_name: str, device: str) -> torch.nn.Module:
    """Loads model class and weights from best checkpoint."""
    ckpt_path = os.path.join(CKPT_DIR, model_name, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train the model first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model_cls, _ = MODEL_MAP[model_name]
    model = model_cls(config=config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"  Loaded {model_name} from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model


def evaluate_model(model_name: str, model, test_loader, device: str) -> dict:
    """Runs Evaluator and saves metrics to results/{model_name}_metrics.json."""
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader)

    # confusion matrix is a tensor/list — convert to list for JSON
    cm = metrics["confusion_matrix"]
    if hasattr(cm, "tolist"):
        metrics["confusion_matrix"] = cm.tolist()

    out_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved -> {out_path}")
    return metrics


def print_metrics(model_name: str, metrics: dict) -> None:
    """Pretty-prints evaluation metrics for one model."""
    print(f"\n{'='*45}")
    print(f"  {model_name.upper()}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"             Human   AI")
    print(f"    Human  [ {cm[0][0]:5d}  {cm[0][1]:5d} ]")
    print(f"    AI     [ {cm[1][0]:5d}  {cm[1][1]:5d} ]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained AI text detection models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "transformer", "bert"],
        default=None,
        help="Which models to evaluate. Defaults to all with existing checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load saved test loader
    test_loader_path = os.path.join(RESULTS_DIR, "test_loader.pt")
    if not os.path.exists(test_loader_path):
        raise FileNotFoundError(f"{test_loader_path} not found. Run train.py first.")

    print(f"Loading test loader from {test_loader_path}...")
    test_loader = torch.load(test_loader_path, map_location=device, weights_only = False)

    # decide which models to evaluate
    if args.models is not None:
        models_to_eval = args.models
    else:
        # auto-detect: evaluate all models that have a checkpoint
        models_to_eval = [
            name for name in ["lstm", "transformer", "bert"]
            if os.path.exists(os.path.join(CKPT_DIR, name, "best.pt"))
        ]
        print(f"Auto-detected checkpoints: {models_to_eval}")

    all_metrics = {}

    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name}...")
        try:
            model = load_model(model_name, device)
            metrics = evaluate_model(model_name, model, test_loader, device)
            print_metrics(model_name, metrics)
            all_metrics[model_name] = metrics
        except FileNotFoundError as e:
            print(f"  Skipping {model_name}: {e}")

    # save combined summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
