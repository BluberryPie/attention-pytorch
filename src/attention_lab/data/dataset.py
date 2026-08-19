from collections.abc import Callable

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from .tokenize import tokenize
from .vocab import EOS, SOS, Vocab


class GigawordDataset(Dataset):
    def __init__(self, data: HFDataset, vocab: Vocab, max_source_len: int = 100):
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.max_source_len = max_source_len

    def __len__(self) -> int:
        return self.data.num_rows

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        source_tokens = tokenize(self.data[idx]["article"])[: self.max_source_len]
        source_ids = self.vocab.encode(source_tokens)
        target_tokens = tokenize(self.data[idx]["summary"])
        target_ids = (
            [self.vocab.stoi[SOS]]
            + self.vocab.encode(target_tokens)
            + [self.vocab.stoi[EOS]]
        )
        return source_ids, target_ids


def _pad_batch(ids_list: list[list[int]], pad_id: int) -> torch.Tensor:
    ids_tensor = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
    ids_tensor_padded = torch.nn.utils.rnn.pad_sequence(
        ids_tensor, padding_value=pad_id, batch_first=True
    )
    return ids_tensor_padded


def make_collate_fn(pad_id: int) -> Callable:
    def collate_fn(
        batch: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids_list, target_ids_list = map(list, zip(*batch))
        return _pad_batch(source_ids_list, pad_id), _pad_batch(target_ids_list, pad_id)

    return collate_fn
