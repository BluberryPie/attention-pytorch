from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    variant: str = "bahdanau"
    vocab_path: str = "vocab"
    checkpoint_path: str = "checkpoints/bahdanau"
    heatmap_path: str = "outputs/heatmap"
    embedding_dim: int = 256
    hidden_dim: int = 512
    attention_dim: int = 512
    batch_size: int = 32
    gradient_clip_norm: float = 5.0
    teacher_forcing_ratio: float = 0.5
    vocab_size: int = 50_000
    subset_size: int = 300_000
    max_source_len: int = 100
    max_target_len: int = 20
    learning_rate: float = 0.001
    num_epochs: int = 10
    num_heatmap_samples: int = 5

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        return cls(**yaml.safe_load(path.read_text()))
