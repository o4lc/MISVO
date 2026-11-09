"""Diversity metric: product of (1 - rep_n/100) for n=2,3,4."""

from nltk import word_tokenize
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def compute_rep_n(text: str, n: int) -> float:
    tokens = word_tokenize(text)
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    rep_n = 100.0 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
    return rep_n


def compute_diversity(text: str) -> float:
    """Compute diversity score: product of (1 - rep_n/100) for n=2,3,4."""
    diversity = 1.0
    for n in range(2, 5):
        rep_n = compute_rep_n(text, n)
        diversity *= 1.0 - rep_n / 100.0
    return diversity


def compute_diversity_batch(responses: list) -> dict:
    """Compute diversity metrics for a batch of responses."""
    diversities = []
    rep2s, rep3s, rep4s = [], [], []
    for resp in responses:
        if len(resp.strip()) == 0:
            resp = " "
        diversities.append(compute_diversity(resp))
        rep2s.append(compute_rep_n(resp, 2))
        rep3s.append(compute_rep_n(resp, 3))
        rep4s.append(compute_rep_n(resp, 4))

    import numpy as np
    return {
        "diversity_mean": float(np.mean(diversities)),
        "diversity_std": float(np.std(diversities)),
        "rep_2": float(np.mean(rep2s)),
        "rep_3": float(np.mean(rep3s)),
        "rep_4": float(np.mean(rep4s)),
    }
