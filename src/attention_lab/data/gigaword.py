from typing import Literal

from datasets import Dataset, load_dataset


Split = Literal["train", "validation", "test"]


def load_gigaword(split: Split) -> Dataset:
    return load_dataset("SalmanFaroz/gigaword", split=split)
