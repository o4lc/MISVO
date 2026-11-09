"""
Best-of-N sampling baseline:
- BoN (top-p): Standard sampling with top-p/temperature, select best by reward

All generation is batched via HuggingFace's generate():
- top-p uses num_return_sequences for parallel sampling
"""

import torch


def generate_bon_topp_responses(
    llm,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_text: str,
    N: int = 1024,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 32,
) -> list:
    """Generate N responses via batched top-p sampling."""
    outputs = llm.generate_samples_batch(
        prompt_ids, attention_mask,
        num_samples=N, batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    prompt_text_decoded = llm.decode(prompt_ids[0])
    full_texts = llm.batch_decode(outputs)
    return [t[len(prompt_text_decoded):] for t in full_texts]


def generate_bon_topp_responses_wave(
    llm,
    prompt_ids_list: list,
    attention_mask_list: list,
    N: int = 1024,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 32,
) -> list:
    """Generate N responses per prompt for P prompts via batched top-p sampling.

    Pools P*N samples into a single generation batch (chunked by batch_size).
    Returns a list of P lists, each containing N response strings (no prompt).
    """
    P = len(prompt_ids_list)
    padded_ids, padded_mask, _ = llm.pad_and_stack_prompts(
        prompt_ids_list, attention_mask_list)
    # Per-prompt decoded prefix for stripping the prompt from responses.
    prompt_decodes = [llm.decode(ids[0]) for ids in prompt_ids_list]

    responses_per_prompt = [[] for _ in range(P)]
    # Chunk over the n_per_prompt axis to keep batch_size in check.
    # batch_size here is total wave-batch (P * n_in_chunk).
    n_per_chunk = max(1, batch_size // max(1, P))
    n_done = 0
    while n_done < N:
        cur_n = min(n_per_chunk, N - n_done)
        outputs = llm.generate_samples_wave(
            padded_ids, padded_mask, n_per_prompt=cur_n,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
        # outputs: (P*cur_n, max_L+T_gen). Decode then strip per prompt.
        full_texts = llm.batch_decode(outputs)
        for row, ft in enumerate(full_texts):
            p = row // cur_n
            responses_per_prompt[p].append(ft[len(prompt_decodes[p]):])
        n_done += cur_n
    return responses_per_prompt
