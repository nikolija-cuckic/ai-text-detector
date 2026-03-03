import sys
sys.path.append(".")

import os
import json
import torch
import numpy as np
from transformers import BertTokenizer

RESULTS_DIR = "results"
ERROR_DIR = "results/errors"
os.makedirs(ERROR_DIR, exist_ok=True)

TOKENIZER_NAME = "bert-base-uncased"

def collect_errors(model, test_loader, device: str) -> list:
    """
    Runs model on test set and collects all misclassified samples.
    Returns:
        list of dicts with keys: input_ids, attention_mask, true_label, pred_label, confidence
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)  # [B, 2]
            preds = probs.argmax(dim=-1)            # [B]

            wrong_mask = preds != labels

            for i in range(len(labels)):
                if wrong_mask[i]:
                    errors.append({
                        "input_ids": input_ids[i].cpu(),
                        "attention_mask": attention_mask[i].cpu(),
                        "true_label": labels[i].item(),
                        "pred_label": preds[i].item(),
                        "confidence": probs[i][preds[i]].item(),
                    })
    return errors


def decode_sample(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> str:
    """Converts token ids back to a readable string, stripping [PAD]."""
    real_len = attention_mask.sum().item()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:real_len].tolist())
    # merge wordpiece tokens (## prefix) into words
    words = []
    for t in tokens:
        if t.startswith("##"):
            if words:
                words[-1] += t[2:]
        else:
            words.append(t)
    return " ".join(words)


def analyze(model, test_loader, device: str, model_name: str, n_show: int = 10) -> dict:
    """
    Saves:
        results/errors/{model_name}_errors.json  - top n_show errors by confidence
        results/errors/{model_name}_error_stats.json - aggregate stats

    Args:
        model: trained model
        test_loader: DataLoader for test set
        device: "cpu" or "cuda"
        model_name: "lstm", "transformer", or "bert"
        n_show: how many errors to include in the JSON report

    Returns:
        stats dict with error counts and rates
    """
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Collecting errors for {model_name}...")
    errors = collect_errors(model, test_loader, device)

    total_errors = len(errors)

    # count by error type
    # false positive: true=human (0), pred=AI (1) - model thinks human text is AI
    # false negative: true=AI (1), pred=human (0) - model misses AI text
    false_positives = [e for e in errors if e["true_label"] == 0 and e["pred_label"] == 1]
    false_negatives = [e for e in errors if e["true_label"] == 1 and e["pred_label"] == 0]

    # count total samples in test set
    total_samples = sum(len(b["labels"]) for b in test_loader)

    stats = {
        "model": model_name,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_samples, 4),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "fp_rate": round(len(false_positives) / max(1, total_samples // 2), 4),
        "fn_rate": round(len(false_negatives) / max(1, total_samples // 2), 4),
        "mean_confidence_on_errors": round(
            float(np.mean([e["confidence"] for e in errors])) if errors else 0.0, 4
        ),
    }

    print(f"Total errors: {total_errors} / {total_samples} ({stats['error_rate']*100:.1f}%)")
    print(f"False positives: {len(false_positives)} (human classified as AI)")
    print(f"False negatives: {len(false_negatives)} (AI classified as human)")
    print(f"Mean confidence on errors: {stats['mean_confidence_on_errors']:.4f}")

    # sort errors by confidence descending
    errors_sorted = sorted(errors, key=lambda x: x["confidence"], reverse=True)

    # decode top n_show errors
    error_records = []
    for e in errors_sorted[:n_show]:
        text = decode_sample(e["input_ids"], e["attention_mask"], tokenizer)
        error_records.append({
            "true_label": "human" if e["true_label"] == 0 else "AI",
            "pred_label": "human" if e["pred_label"] == 0 else "AI",
            "confidence": round(e["confidence"], 4),
            "text_preview": text[:300],  # first 300 chars
        })

    # save reports
    stats_path = os.path.join(ERROR_DIR, f"{model_name}_error_stats.json")
    errors_path = os.path.join(ERROR_DIR, f"{model_name}_errors.json")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(error_records, f, indent=2, ensure_ascii=False)

    print(f"  Stats  saved -> {stats_path}")
    print(f"  Errors saved -> {errors_path}")

    return stats
