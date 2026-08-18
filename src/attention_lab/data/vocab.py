from __future__ import annotations

from collections import Counter
from itertools import chain


PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos: list[str] = itos
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(itos)}

    @classmethod
    def build(cls, tokenized_data: list[list[str]], max_size: int = 50_000) -> Vocab:
        counter: Counter = Counter(chain.from_iterable(tokenized_data))
        most_common_tokens: list = counter.most_common(max_size - len(SPECIAL_TOKENS))
        itos = SPECIAL_TOKENS + [token for token, _ in most_common_tokens]
        return cls(itos)        
