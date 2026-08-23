from collections import defaultdict

from rouge_score.rouge_scorer import RougeScorer


def score_pair(generated: str, reference: str, scorer: RougeScorer) -> dict[str, float]:
    """Score a single (generated, reference pair) and return that pair's F1 scores"""
    scores = scorer.score(reference, generated)  # Ground truth comes first
    f1_scores = {k: v.fmeasure for k, v in scores.items()}
    return f1_scores


def average_rouge(
    generated_list: list[str], reference_list: list[str], scorer: RougeScorer
) -> dict[str, float]:
    accumulated_f1_scores = defaultdict(float)
    for generated, reference in zip(generated_list, reference_list):
        f1_scores = score_pair(generated, reference, scorer)
        for k, v in f1_scores.items():
            accumulated_f1_scores[k] += v
    averaged_f1_scores = {
        k: v / len(generated_list) for k, v in accumulated_f1_scores.items()
    }
    return averaged_f1_scores
