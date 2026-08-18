import itertools

import pytest

from attention_lab.data.tokenize import tokenize
from attention_lab.data.vocab import EOS, PAD, SOS, UNK, Vocab


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
