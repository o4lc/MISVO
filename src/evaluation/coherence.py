"""Coherence metric: cosine similarity between prompt and response via SimCSE."""

import numpy as np


def compute_coherence(prompts: list, responses: list) -> float:
    """Compute average coherence (cosine similarity) using SimCSE."""
    from simcse import SimCSE
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

    # Handle empty responses
    cleaned_responses = [r if len(r.strip()) > 0 else " " for r in responses]

    similarities = np.array(model.similarity(prompts, cleaned_responses))
    # Diagonal gives prompt_i <-> response_i similarity
    return float(similarities.trace() / len(similarities))


def compute_coherence_batch(prompts: list, responses: list, batch_size: int = 64) -> dict:
    """Compute coherence in batches to avoid OOM."""
    from simcse import SimCSE
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

    all_sims = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = [r if len(r.strip()) > 0 else " " for r in responses[i:i + batch_size]]
        sims = np.array(model.similarity(batch_prompts, batch_responses))
        for j in range(len(batch_prompts)):
            all_sims.append(sims[j, j])

    return {
        "coherence_mean": float(np.mean(all_sims)),
        "coherence_std": float(np.std(all_sims)),
    }
