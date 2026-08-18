import pytest

from attention_lab.data.tokenize import tokenize


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
