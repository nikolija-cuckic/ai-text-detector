from dataclasses import dataclass, field


# Data config
@dataclass
class DataConfig:
    data_path: str = "data/hc3_all.jsonl"
    max_len: int = 256          # use 512 for final training
    batch_size: int = 32
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0        # 0 = safe on Windows, 2-4 on Linux/Colab
    seed: int = 9


# Transformer classifier config

@dataclass
class TransformerConfig:
    # data (must match DataConfig)
    vocab_size: int = 30522     # BertTokenizer vocab size
    max_len: int = 256
    num_classes: int = 2

    # architecture
    d_model: int = 128
    n_heads: int = 8            # d_model must be divisible by n_heads
    n_layers: int = 4
    dropout: float = 0.1
    bias: bool = False

    # training
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 20
    patience: int = 3           # early stopping patience
    seed: int = 9


# LSTM baseline config

@dataclass
class LSTMConfig:
    # data
    vocab_size: int = 30522
    num_classes: int = 2

    # architecture
    lstm_embed_dim: int = 128
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    dropout: float = 0.1

    # training
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 20
    patience: int = 3
    seed: int = 9


# BERT fine-tuning config 

@dataclass
class BERTConfig:
    # pretrained model
    bert_model_name: str = "bert-base-uncased"
    num_classes: int = 2

    # head
    dropout: float = 0.1

    # training — lower lr than scratch models, BERT weights are pretrained
    lr: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 5         # BERT fine-tuning needs fewer epochs
    patience: int = 2
    seed: int = 9


#  Quick sanity check

if __name__ == "__main__":
    data_cfg = DataConfig()
    trans_cfg = TransformerConfig()
    lstm_cfg = LSTMConfig()
    bert_cfg = BERTConfig()

    print("DataConfig:")
    print(f"  data_path  : {data_cfg.data_path}")
    print(f"  max_len    : {data_cfg.max_len}")
    print(f"  batch_size : {data_cfg.batch_size}")
    print(f"  seed       : {data_cfg.seed}")

    print("\nTransformerConfig:")
    print(f"  d_model    : {trans_cfg.d_model}")
    print(f"  n_heads    : {trans_cfg.n_heads}")
    print(f"  n_layers   : {trans_cfg.n_layers}")
    print(f"  lr         : {trans_cfg.lr}")

    print("\nLSTMConfig:")
    print(f"  embed_dim  : {lstm_cfg.lstm_embed_dim}")
    print(f"  hidden_dim : {lstm_cfg.lstm_hidden_dim}")
    print(f"  num_layers : {lstm_cfg.lstm_num_layers}")
    print(f"  lr         : {lstm_cfg.lr}")

    print("\nBERTConfig:")
    print(f"  model_name : {bert_cfg.bert_model_name}")
    print(f"  lr         : {bert_cfg.lr}")
    print(f"  max_epochs : {bert_cfg.max_epochs}")
    print(f"  patience   : {bert_cfg.patience}")
