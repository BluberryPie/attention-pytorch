import itertools
import json

import pytest
from datasets import Dataset as HFDataset

from attention_lab.data.dataset import GigawordDataset
from attention_lab.data.tokenize import tokenize
from attention_lab.data.vocab import EOS, PAD, SOS, SPECIAL_TOKENS, UNK, Vocab


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("I have a dream . ", ["I", "have", "a", "dream", "."]),
        ("This  too shall pass .", ["This", "too", "shall", "pass", "."]),
    ],
    ids=["collapses_leading_trailing_space", "keeps_multiple_spaces_collapsed"],
)
def test_tokenize_splits_on_whitespaces(raw: str, expected: list[str]):
    assert tokenize(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "Acme corp. signed a new contract .",
            ["Acme", "corp.", "signed", "a", "new", "contract", "."],
        ),
    ],
    ids=["keeps_abbreviation_period_attached"],
)
def test_tokenize_preserves_abbreviations(raw: str, expected: list[str]):
    assert tokenize(raw) == expected


def test_vocab_init_builds_stoi_from_itos():
    itos: list[str] = ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    for idx, token in enumerate(itos):
        assert vocab.stoi[token] == idx


def test_vocab_build_keeps_special_token_ids_fixed():
    expected = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
    tokenized_data = [
        ["The", "cat", "sat", "on", "the", "mat"],
        ["Hit", "the", "rat", "with", "a", "bat"],
    ]
    vocab = Vocab.build(tokenized_data)
    for special_token, idx in expected.items():
        assert vocab.stoi[special_token] == idx


def test_vocab_build_orders_tokens_by_frequency():
    tokenized_data = [
        ["the", "cat", "saw", "the", "cat"],
        ["the", "cat", "saw", "a", "rat"],
        ["the", "saw", "saw", "a", "saw"],
    ]
    vocab = Vocab.build(tokenized_data)
    # saw: 5 / the: 4 / cat: 3 / a: 2 / rat: 1
    tokens_ordered = ["saw", "the", "cat", "a", "rat"]
    for t_1, t_2 in itertools.pairwise(tokens_ordered):
        assert vocab.stoi[t_1] < vocab.stoi[t_2]


def test_vocab_build_caps_at_max_size():
    max_size = 8
    tokenized_data = [
        ["the", "cat", "saw", "the", "cat"],
        ["the", "cat", "saw", "a", "rat"],
        ["the", "saw", "saw", "a", "saw"],
    ]
    # saw: 5 / the: 4 / cat: 3 / a: 2 / rat: 1
    # 'rat' should be excluded
    vocab = Vocab.build(tokenized_data, max_size=max_size)
    assert len(vocab.itos) == max_size
    assert "rat" not in vocab.stoi


def test_vocab_does_not_pad_with_small_inputs():
    max_size = 100
    tokenized_data = [
        ["I", "have", "a", "dream", "."],
        ["This", "too", "shall", "pass", "."],
    ]
    # Vocab size should be 13 (4 special tokens + 9 from input)
    vocab = Vocab.build(tokenized_data, max_size=max_size)
    assert len(vocab.itos) == 13


def test_vocab_build_deduplicates_across_sublists():
    tokenized_data = [
        ["the", "cat", "saw", "the", "cat"],
        ["the", "cat", "saw", "a", "rat"],
        ["the", "saw", "saw", "a", "saw"],
    ]
    # Vocab size should be 9 (4 special tokens + 5 from input)
    vocab = Vocab.build(tokenized_data)
    assert len(vocab.itos) == 9


def test_vocab_encode_maps_known_tokens_to_ids():
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    # "I": 4, "have": 5, "a": 6, "dream": 7, ".": 8
    tokens = ["a", "dream", "have", "I", "."]
    ids = vocab.encode(tokens)
    assert ids == [6, 7, 5, 4, 8]


def test_vocab_encode_maps_unknown_tokens_to_unk():
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    # "I": 4, "have": 5, "a": 6, "dream": 7, ".": 8
    tokens = ["Cats", "have", "dreams", "."]
    ids = vocab.encode(tokens)
    unk = vocab.stoi[UNK]
    assert ids == [unk, 5, unk, 8]


def test_vocab_decode_maps_ids_to_tokens():
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    # "I": 4, "have": 5, "a": 6, "dream": 7, ".": 8
    pad, unk, sos, eos = (
        vocab.stoi[PAD],
        vocab.stoi[UNK],
        vocab.stoi[SOS],
        vocab.stoi[EOS],
    )
    ids = [sos, unk, 5, unk, 8, pad, pad, eos]
    tokens = vocab.decode(ids)
    assert tokens == [SOS, UNK, "have", UNK, ".", PAD, PAD, EOS]


def test_vocab_decode_raises_on_out_of_range_id():
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    # "I": 4, "have": 5, "a": 6, "dream": 7, ".": 8
    ids = [4, 5, 6, 9]  # id=9 doesn't exist
    with pytest.raises(IndexError):
        vocab.decode(ids)


def test_vocab_save_writes_itos_to_file(tmp_path):
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    file_path = tmp_path / "test.json"
    vocab.save(path=file_path)
    itos_loaded = json.loads(file_path.read_text(encoding="utf-8"))
    assert itos == itos_loaded


def test_vocab_load_reconstructs_vocab_from_file(tmp_path):
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(itos))
    vocab = Vocab.load(file_path)
    assert itos == vocab.itos


def test_vocab_size_returns_number_of_tokens():
    itos: list[str] = SPECIAL_TOKENS + ["I", "have", "a", "dream", "."]
    vocab = Vocab(itos)
    assert vocab.size == len(itos)


def test_gigaword_dataset_len_matches_row_count():
    data: HFDataset = HFDataset.from_dict(
        {"article": ["A", "B", "C"], "summary": ["X", "Y", "Z"]}
    )
    vocab = Vocab(SPECIAL_TOKENS + ["A", "B", "C", "X", "Y", "Z"])
    dataset = GigawordDataset(data, vocab)
    assert len(dataset) == 3


@pytest.fixture
def sample_data() -> HFDataset:
    return HFDataset.from_dict(
        {
            "article": ["the cat sat on the old mat quitely"],
            "summary": ["cat sat quitely"],
        }
    )


@pytest.fixture
def sample_vocab() -> Vocab:
    return Vocab(SPECIAL_TOKENS + ["the", "cat", "sat", "on", "old", "mat", "quitely"])


def test_gigaword_dataset_getitem_encodes_source(sample_data, sample_vocab):
    dataset = GigawordDataset(data=sample_data, vocab=sample_vocab)
    source_ids, _ = dataset[0]
    # the: 4 / cat: 5 / sat: 6 / on: 7 / old: 8 / mat: 9 / quitely: 10
    # the cat sat on the old mat quitely => [4, 5, 6, 7, 4, 8, 9, 10]
    assert source_ids == [4, 5, 6, 7, 4, 8, 9, 10]


def test_gigaword_dataset_getitem_truncates_source(sample_data, sample_vocab):
    max_source_len = 4
    dataset = GigawordDataset(
        data=sample_data, vocab=sample_vocab, max_source_len=max_source_len
    )
    source_ids, _ = dataset[0]
    assert source_ids == [4, 5, 6, 7]


def test_gigaword_dataset_getitem_wraps_target_with_sos_eos(sample_data, sample_vocab):
    dataset = GigawordDataset(data=sample_data, vocab=sample_vocab)
    _, target_ids = dataset[0]
    # the: 4 / cat: 5 / sat: 6 / on: 7 / old: 8 / mat: 9 / quitely: 10
    # cat sat quitely => [5, 6, 10]
    sos_id, eos_id = sample_vocab.stoi[SOS], sample_vocab.stoi[EOS]
    assert target_ids == [sos_id] + [5, 6, 10] + [eos_id]
