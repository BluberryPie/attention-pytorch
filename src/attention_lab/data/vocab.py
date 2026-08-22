from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos: list[str] = itos
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    @classmethod
    def build(
        cls, tokenized_data: Iterable[list[str]], max_size: int = 50_000
    ) -> Vocab:
        counter: Counter = Counter(chain.from_iterable(tokenized_data))
        most_common_tokens: list = counter.most_common(max_size - len(SPECIAL_TOKENS))
        itos = SPECIAL_TOKENS + [token for token, _ in most_common_tokens]
        return cls(itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.stoi[UNK]) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[_id] for _id in ids]

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, indent=4)

    @classmethod
    def load(cls, path: Path) -> Vocab:
        with open(path, "r", encoding="utf-8") as f:
            itos = json.load(f)
        return cls(itos)
