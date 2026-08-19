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
