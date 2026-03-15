# AI Text Detector

A comparative study of Transformer and LSTM architectures for detecting AI-generated text, with a focus on architectural trade-offs, compute budgets, and interpretability.

---

## Overview

This project implements and compares three model families for binary classification of human-written vs. AI-generated text (ChatGPT), trained on the [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3):

| Model | Parameters | Description |
|---|---|---|
| **TransformerClassifier** | ~2.5 M | Encoder-only Transformer built from scratch (4 layers, d_model=128, 8 heads) |
| **LSTMClassifier** | ~1.5 M | Bidirectional LSTM baseline with packed-sequence support |
| **BERTClassifier** | ~110 M | `bert-base-uncased` fine-tuned with a custom classification head; supports frozen and full fine-tuning modes |

All models share the same training loop, data pipeline, and evaluation interface, enabling direct and fair comparison under identical conditions.

---

## Motivation

The project grew out of a practical question: *how much does architecture matter when you fix the compute budget?*

A small Transformer trained from scratch is orders of magnitude cheaper than BERT, yet operates on the same input format and is trained the same way. By holding the training procedure constant and varying only the architecture, we can isolate the effect of model depth, parameter count, and inductive biases on classification accuracy.

This connects directly to broader questions in language model research — specifically around efficient architectures, recurrent vs. parallel computation, and the accuracy-vs-compute trade-off — topics at the heart of current research on reasoning with small models.

---

## Project Structure

```
ai-text-detector/
├── configs/
│   └── config.py           # Dataclass configs for all models and data pipeline
├── data/
│   └── dataset.py          # HC3 loading, cleaning, tokenization, DataLoader construction
├── layers/
│   ├── attention.py        # Bidirectional multi-head self-attention (manual implementation)
│   ├── embeddings.py       # Token + sinusoidal positional embeddings
│   ├── encoder_block.py    # Pre-norm Transformer encoder block (MHA + FFN + residuals)
│   └── feedforward.py      # Position-wise feed-forward network
├── models/
│   ├── classifier.py       # TransformerClassifier (scratch encoder + CLS head)
│   ├── bert_classifier.py  # BERTClassifier (HuggingFace BERT + custom head)
│   └── lstm_baseline.py    # Bidirectional LSTM classifier
├── training/
│   ├── trainer.py          # Training loop: AdamW, cosine LR schedule, early stopping, checkpointing
│   └── evaluator.py        # Test-set evaluation: accuracy, precision, recall, F1, confusion matrix
├── evaluating/
│   ├── attention_viz.py    # Attention heatmap visualizations (per-layer, per-head)
│   └── error_analysis.py   # False positive/negative analysis with confidence scores
├── scripts/
│   ├── train.py            # CLI entry point for training (supports --models, --lr, --max_epochs, ...)
│   └── evaluate.py         # CLI entry point for evaluation and analysis
└── results/                # Training histories (JSON), attention plots, error reports
```

---

## Implementation Details

### Transformer (from scratch)

The small Transformer is implemented entirely without any framework-level attention utilities:

- **Attention** — scaled dot-product multi-head attention with manual Q/K/V projections; padding mask applied before softmax (fills with `-inf`) with NaN protection for fully-padded rows
- **Positional encoding** — sinusoidal, fixed (no learned positions)
- **Architecture** — pre-LayerNorm blocks; final LayerNorm before the classification head; CLS token at position 0 as the sequence representation
- **Initialization** — Normal(0, 0.02) for Linear and Embedding layers; Orthogonal init for LSTM weights

### LSTLM baseline

Bidirectional LSTM with `pack_padded_sequence` / `pad_packed_sequence` so that padding tokens are never processed. The final hidden states from both directions are concatenated and passed through a linear head.

### BERT fine-tuning

Two modes are supported:
- **Frozen** — only the classification head is trained; exposes the effect of pure representation quality
- **Full** — all parameters are updated; uses a lower learning rate (2e-5 vs 1e-3) appropriate for fine-tuning pretrained weights

Attention weights are retrieved via `output_attentions=True` and exposed in the same interface as the scratch Transformer, enabling cross-model attention comparison.

### Training infrastructure

- **Optimizer** — AdamW with weight decay
- **Schedule** — Cosine annealing over the full training budget
- **Gradient clipping** — norm clipped to 1.0
- **Early stopping** — based on validation accuracy; patience configurable per model
- **Checkpointing** — `best.pt` (highest val accuracy) and `last.pt` (for resuming interrupted runs) are saved automatically
- **History** — training/validation loss and accuracy per epoch serialized to JSON for offline plotting

---

## Experiments and Analysis

Beyond basic training, the project includes tooling for:

### Attention Visualization (`evaluating/attention_viz.py`)

- Extracts attention weight tensors from forward passes
- Plots mean attention across heads per layer (heatmap)
- Plots all attention heads from a given layer in a grid
- Compares attention patterns between a human-written and an AI-generated sample

### Error Analysis (`evaluating/error_analysis.py`)

- Collects all misclassified test samples with their predicted label and model confidence
- Reports false positive rate (human → AI) and false negative rate (AI → human) separately
- Saves the top-*n* highest-confidence errors with decoded text snippets for qualitative inspection
- Outputs structured JSON reports for reproducible analysis

---

## Dataset

**HC3** (Human ChatGPT Comparison Corpus) — a publicly available dataset of question-answer pairs where human and ChatGPT answers are provided for the same questions across multiple domains (medicine, finance, open-domain QA, etc.).

- Download: `data/hc3_all.jsonl` from [Hugging Face](https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl)
- Labels: `0` = human, `1` = AI (ChatGPT)
- Tokenization: `bert-base-uncased` WordPiece tokenizer (vocab size 30,522); max sequence length 256
- Splits: row-level train/test split (90/10), with a further val split (10%) taken from the training set for early stopping

---

## Setup

```bash
git clone https://github.com/nikolija-cuckic/ai-text-detector
cd ai-text-detector

pip install -r requirements.txt

# Download the dataset
mkdir data
curl -L https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl -o data/hc3_all.jsonl
```

**Requirements:** `torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Usage

### Training

```bash
# Train a specific model
python scripts/train.py --models transformer
python scripts/train.py --models lstm
python scripts/train.py --models bert

# Train all models with custom hyperparameters
python scripts/train.py --models lstm transformer bert --max_epochs 15 --lr 5e-4

# Train the small Transformer with a shorter context window
python scripts/train.py --models transformer --max_len 128 --batch_size 64
```

### Evaluation

```bash
python scripts/evaluate.py --model transformer
python scripts/evaluate.py --model bert --analyze --visualize
```

Results (confusion matrix, F1, attention plots, error reports) are saved under `results/`.

---

## Key Design Decisions

**Shared interface.** All models expose the same `forward(input_ids, attention_mask) → (logits, attentions)` signature. The LSTM returns `None` for attentions. This lets the trainer, evaluator, and visualization code work without any model-specific branching.

**Row-level train/test split.** HC3 is split at the question level (not the answer level), ensuring that the model never sees a test question's phrasing during training, even if that question appears in a different context.

**Small Transformer design choices.** `d_model=128`, 4 layers, 8 heads — keeping the parameter count in the low millions to enable training on CPU within a reasonable time budget, and to make the accuracy-vs-compute comparison with BERT meaningful.

**Parameter count tracking.** All models implement `count_parameters(trainable_only=True)`, which is printed at the start of every training run for reproducibility.

---

## Results Summary

Experiments were run with two sequence length settings (`max_len=128` and `max_len=256`). All models use the same training loop (AdamW, cosine schedule, early stopping on validation accuracy).

### max_len = 128

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | Trainable Params |
|---|---|---|---|---|---|---|
| LSTM (BiLSTM) | **0.9794** | **0.9672** | 0.9682 | 0.9661 | **0.9964** | ~1.5 M |
| Transformer (scratch) | 0.9775 | 0.9644 | 0.9594 | **0.9694** | 0.9970 | ~2.5 M |
| BERT (frozen head) | 0.9092 | 0.8670 | 0.8045 | 0.9400 | 0.9714 | ~0.6 K |

### max_len = 256

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | Trainable Params |
|---|---|---|---|---|---|---|
| LSTM (BiLSTM) | **0.9864** | **0.9783** | **0.9823** | 0.9743 | **0.9983** | ~1.5 M |
| Transformer (scratch) | 0.9846 | 0.9755 | 0.9797 | 0.9713 | 0.9978 | ~2.5 M |
| BERT (frozen head) | 0.9198 | 0.8820 | 0.8209 | **0.9530** | 0.9790 | ~0.6 K |

### Key observations

- The small Transformer (2.5 M parameters, trained from scratch) **matches BERT-frozen** (110 M parameter pretrained model, head-only training) despite using ~44x fewer parameters, achieving 98.5% test accuracy vs. 92.0%.
- The BiLSTM achieves **comparable accuracy to the scratch Transformer** with fewer parameters and a sequential inductive bias, raising the question of whether attention is necessary at this scale.
- Longer context (`max_len=256`) consistently improves all models, suggesting that AI-generated text has stylistic patterns that extend across longer sequences.

---

## Relevance to Research on Reasoning with Small Models

This project directly addresses several questions that arise in research on compute-efficient language models:

- **Architecture vs. scale** — a small Transformer trained from scratch vs. a large pretrained model: when does scale win, and when does architecture? The results show that at small parameter count, a scratch Transformer approaches BERT-frozen performance.
- **Compute budget awareness** — comparing models under a fixed training budget (same epochs, same data, same hardware) makes the accuracy-vs-compute trade-off explicit. A LSTM with 1.5 M parameters trained on CPU converges in minutes and achieves 98.6% accuracy, while BERT (frozen) takes similar time but only reaches 92%. Full fine-tuning of BERT closes this gap but at 70x the parameter cost.
- **Recurrent inductive bias** — the LSTM baseline provides a concrete reference point for evaluating whether Transformer attention is worth the parameter and compute cost at small scale.
- **Interpretability** — attention visualization enables inspection of what linguistic patterns each architecture focuses on, which is a useful diagnostic when analysing model behaviour under capacity constraints.

---
