from .bahdanau import BahdanauAttention
from .base import AttentionBase


def get_attention(name: str, **kwargs) -> AttentionBase:
    if name == "bahdanau":
        return BahdanauAttention(**kwargs)
    else:
        raise ValueError(f"name [{name}] is not supported")
