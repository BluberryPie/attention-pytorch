import pytest
from rouge_score.rouge_scorer import RougeScorer

from attention_lab.eval.rouge import average_rouge, score_pair


@pytest.fixture
def rouge_scorer() -> RougeScorer:
    return RougeScorer(rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@pytest.mark.parametrize(
    "generated, reference",
    [
        ("The cat sat on the mat .", "The cat sat on the mat ."),
        ("I have a dream .", "I have a dream ."),
    ],
)
def test_score_pair_identical_strings_gives_perfect_scores(
    rouge_scorer, generated, reference
):
    scores = score_pair(generated, reference, rouge_scorer)
    for score in scores.values():
        assert score == 1.0


def test_score_pair_completely_different_strings_gives_zero_scores(rouge_scorer):
    generated = "The cat sat on the mat"
    reference = "I have a dream"
    scores = score_pair(generated, reference, rouge_scorer)
    for score in scores.values():
        assert score == 0.0


def test_average_rouge_matches_manual_average(rouge_scorer):
    generated_list = [
        "The cat sat on the mat",
        "I have a dream",
        "I think, therefore I am",
    ]
    reference_list = [
        "The cat ran over the mat",
        "I had a dream",
        "I think, so I exist",
    ]
    accumulated_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for generated, reference in zip(generated_list, reference_list):
        scores = score_pair(generated, reference, rouge_scorer)
        for k, v in scores.items():
            accumulated_scores[k] += v

    average_scores = {k: (v / 3) for k, v in accumulated_scores.items()}
    # Make sure to use pytest.approx since we're comparing floats
    assert pytest.approx(average_scores) == average_rouge(
        generated_list, reference_list, rouge_scorer
    )
