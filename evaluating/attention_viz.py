import sys
sys.path.append(".")

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import BertTokenizer

RESULTS_DIR = "results/attention"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOKENIZER_NAME = "bert-base-uncased"


def get_attention_weights(model, batch: dict, device: str, model_name: str) -> list:
    """
    Runs one forward pass and returns attention weights.
    TransformerClassifier returns list of [B, n_heads, seq_len, seq_len].
    BERTClassifier returns tuple of [B, n_heads, seq_len, seq_len].
    LSTMClassifier returns None.
    """
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        _, attentions = model(input_ids, attention_mask)

    if attentions is None:
        raise ValueError(f"{model_name} doesn't have attention weights (LSTM).")

    # normalize to list of tensors on cpu
    return [a.cpu() for a in attentions]


def plot_attention_head(attn_matrix: np.ndarray, tokens: list, title: str, save_path: str) -> None:
    """
    Plots a single attention head as a heatmap.

    attn_matrix: [seq_len, seq_len] numpy array
    tokens: list of token strings (length = seq_len)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=attn_matrix.max())

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)

    ax.set_xlabel("Key tokens (attended to)")
    ax.set_ylabel("Query tokens (attending from)")
    ax.set_title(title, fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_mean_attention(attentions: list, tokens: list, model_name: str, sample_idx: int, label: int) -> None:
    """
    For each layer, plots the mean attention across all heads for one sample.
    Saves one image per layer.
    attentions: list of [B, n_heads, seq_len, seq_len] tensors
    """
    label_str = "human" if label == 0 else "AI"
    seq_len = len(tokens)

    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: [B, n_heads, seq_len, seq_len]
        # take sample_idx, mean over heads, clip to actual tokens
        attn = layer_attn[sample_idx].mean(dim=0).numpy()  # [seq_len, seq_len]
        attn = attn[:seq_len, :seq_len]

        title = f"{model_name} | layer {layer_idx + 1} | mean over heads | {label_str}"
        fname = f"{model_name}_sample{sample_idx}_layer{layer_idx + 1:02d}_{label_str}.png"
        save_path = os.path.join(RESULTS_DIR, fname)

        plot_attention_head(attn, tokens, title, save_path)
        print(f"  Saved {save_path}")


def plot_all_heads(attentions: list, tokens: list, model_name: str, sample_idx: int, label: int, layer_idx: int = 0) -> None:
    """
    Plots all attention heads from one layer in a grid.
    attentions: list of [B, n_heads, seq_len, seq_len]
    layer_idx: which layer to visualize
    """
    label_str = "human" if label == 0 else "AI"
    layer_attn = attentions[layer_idx][sample_idx]  # [n_heads, seq_len, seq_len]
    n_heads = layer_attn.shape[0]
    seq_len = len(tokens)

    cols = 4
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for h in range(n_heads):
        attn = layer_attn[h].numpy()[:seq_len, :seq_len]
        ax = axes[h]
        ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=attn.max())
        ax.set_title(f"Head {h + 1}", fontsize=8)
        ax.set_xticks(range(0, seq_len, max(1, seq_len // 8)))
        ax.set_yticks(range(0, seq_len, max(1, seq_len // 8)))
        ax.tick_params(labelsize=6)

    # hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"{model_name} | layer {layer_idx + 1} | all heads | {label_str}",
        fontsize=11
    )
    plt.tight_layout()

    fname = f"{model_name}_sample{sample_idx}_layer{layer_idx + 1:02d}_allheads_{label_str}.png"
    save_path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved {save_path}")


def visualize(model, test_loader, device: str, model_name: str, n_samples: int = 2) -> None:
    """
    Picks n_samples from test set (one human, one AI if possible) and generates attention plots.

    Args:
    model: trained TransformerClassifier or BERTClassifier
    test_loader: DataLoader for test set
    device: "cpu" or "cuda"
    model_name: "transformer" or "bert"
    n_samples: how many samples to visualize
    """
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    # collect one human and one AI sample
    samples = []
    for batch in test_loader:
        for i in range(batch["input_ids"].shape[0]):
            label = batch["labels"][i].item()
            if any(s["label"] == label for s in samples):
                continue  # already have this class
            samples.append({
                "input_ids": batch["input_ids"][i].unsqueeze(0),
                "attention_mask": batch["attention_mask"][i].unsqueeze(0),
                "label": label,
            })
            if len(samples) == n_samples:
                break
        if len(samples) == n_samples:
            break

    for sample_idx, sample in enumerate(samples):
        single_batch = {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
        }
        label = sample["label"]

        attentions = get_attention_weights(model, single_batch, device, model_name)

        # decode tokens, strip [PAD]
        ids = sample["input_ids"][0].tolist()
        mask = sample["attention_mask"][0].tolist()
        real_len = sum(mask)
        tokens = tokenizer.convert_ids_to_tokens(ids[:real_len])

        # cap at 40 tokens
        tokens = tokens[:40]

        print(f"\nSample {sample_idx} | label={'human' if label == 0 else 'AI'} | {len(tokens)} tokens")

        # mean attention per layer
        plot_mean_attention(attentions, tokens, model_name, 0, label)

        # all heads from first layer
        plot_all_heads(attentions, tokens, model_name, 0, label, layer_idx=0)
