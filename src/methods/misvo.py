"""
MISVO: Minimally Invasive Steering Vector Optimization

Test-time alignment via policy gradient (REINFORCE) with regularization
in pre-logit space. Finds steering vectors u_t that maximize reward while
minimally disrupting the base model's distribution.

Supported regularizers (paper notation in brackets):
  "frozen_fisher" (MISVO) [frozen-reference Fisher F̄_t]:
    - Base distributions are bootstrapped from Phase 1's step-0 rollouts
      (u=0 → already base-policy); stores per-trajectory base pre-logits
      (n_samples, T, d).
    - F_t = (1/n_samples) Σ_i Var_{p_{t,i}}(W u_t)  (average of per-trajectory Fishers)
    - Per-chunk recomputes p_{t,i} via one extra lm_head pass on (n_samples, c, d)
    - Fisher reg gradient: λ W^T(p_t ⊙ v_t - p_t · dot(p_t, v_t)), v_t = W u_t
  "steered_traj_fisher" [on-policy rollouts, reference Fisher F̄_t^u]:
    - Uses per-trajectory pre-logits z_t^i of the current (steered) rollouts
    - Computes p_base^i = softmax(W z_t^i) (unperturbed) for the Fisher
    - Fisher vectors averaged across valid trajectories at each position
  "fully_adaptive_fisher" [on-policy rollouts, on-policy Fisher F̃_t^u]:
    - Reuses p_batch = softmax(W(z_t^i + u_t)) already computed for the
      REINFORCE gradient — no extra lm_head forward pass
  "l2": simple u_t^T u_t norm penalty (hidden-space ℓ2)
    - L2 reg gradient: λ u_t
  "kl": KL divergence penalty folded into the REINFORCE reward (direct KL)
    - R_kl^i = R^i - λ Σ_t [log p_u(y_t^i) - log p_0(y_t^i)]
    - The modified reward enters REINFORCE as the advantage; gradient
      is plain REINFORCE with no extra penalty gradient term
Unregularized steering is any regularizer with λ = 0.

Key equations:
  Perturbed distribution: p_u(y_t) = softmax(W(z_t + u_t) + b)
  Objective: max E[R(x,y)] - (λ/2) Σ_t u_t^T R_t u_t  (R_t = F_t or I)
  REINFORCE gradient: E[(R - baseline) W^T(e_{y_t} - p_{u,t})]
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import logger
from src.methods.aisp import RewardNormalizer, kl_deviation


def _safe_batched_sampling(
    llm, prompt_ids, attention_mask, perturbation,
    n_samples, max_new_tokens, temperature, top_p,
    prompt_kv_cache=None,
):
    """Batched sampling generation with sequential batch=1 fallback.

    Tries the single batched call first. On a non-OOM RuntimeError (e.g.
    Gemma3 hybrid-attention shape mismatches on long prompts), falls back
    to n_samples sequential batch=1 calls and concatenates the results.
    OOM propagates so callers can re-chunk. If the batch=1 fallback also
    raises, the error is re-raised.

    Returns (output_ids, pre_logits) with the same shapes as the batched
    call.
    """
    try:
        return llm.generate_with_perturbation_sampling(
            prompt_ids, attention_mask, perturbation,
            n_samples, max_new_tokens, temperature, top_p,
            prompt_kv_cache=prompt_kv_cache,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        # Only fall back for the known Gemma3 hybrid-attention shape mismatch.
        # OOM, CUDA assert, and other RuntimeErrors must propagate so the
        # caller's chunk-retry / OOM-retry logic can act on them.
        # Patterns:
        #   SDPA: "The expanded size of the tensor (X) must match the existing size (Y)"
        #   Eager: "The size of tensor a (X) must match the size of tensor b (Y)"
        is_shape_mismatch = (
            "expanded size of the tensor" in msg
            or "must match the existing size" in msg
            or "must match the size of tensor" in msg
        )
        if not is_shape_mismatch:
            raise
        logger.warning(
            f"  Batched generation hit shape-mismatch (n={n_samples}): {e}. "
            f"Falling back to {n_samples} sequential batch=1 calls.")
        torch.cuda.empty_cache()
        ids_list, prelogs_list = [], []
        for i in range(n_samples):
            ids_i, prelogs_i = llm.generate_with_perturbation_sampling(
                prompt_ids, attention_mask, perturbation,
                1, max_new_tokens, temperature, top_p,
                prompt_kv_cache=prompt_kv_cache,
            )
            ids_list.append(ids_i)
            prelogs_list.append(prelogs_i)
        max_len = max(o.shape[1] for o in ids_list)
        for j, o in enumerate(ids_list):
            if o.shape[1] < max_len:
                pad = torch.full((o.shape[0], max_len - o.shape[1]),
                                 llm.tokenizer.pad_token_id,
                                 dtype=o.dtype, device=o.device)
                ids_list[j] = torch.cat([o, pad], dim=1)
        max_T = max(p.shape[1] for p in prelogs_list)
        for j, p in enumerate(prelogs_list):
            if p.shape[1] < max_T:
                pad = torch.zeros(p.shape[0], max_T - p.shape[1], p.shape[2],
                                  dtype=p.dtype)
                prelogs_list[j] = torch.cat([p, pad], dim=1)
        return torch.cat(ids_list, dim=0), torch.cat(prelogs_list, dim=0)


def run_misvo(
    llm,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reward_fn,
    prompt_text: str,
    lam: float = 1.0,
    n_samples: int = 16,
    n_steps: int = 20,
    lr: float = 0.01,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
    reward_fn_batch=None,
    reward_normalizer: "RewardNormalizer | None" = None,
    regularizer: str = "frozen_fisher",
    binary_reward_fn=None,
    n_chunks: int = 4,
    gen_chunks: int = 1,
    track_diagnostics: bool = True,
) -> dict:
    """Run MISVO on a single prompt.

    Phase 1: REINFORCE + regularization gradient ascent to optimize u. At
             step 0, base distributions are bootstrapped from the rollouts
             themselves (u=0 → base policy; Fisher gradient is 0 anyway).
    Phase 2: Final greedy generation with optimized u.

    Args:
        reward_fn_batch: optional callable(prompt, [responses]) -> [float] for batched scoring
        reward_normalizer: optional RewardNormalizer for stabilizing REINFORCE advantages
        regularizer: "frozen_fisher" (default; MISVO), "steered_traj_fisher",
            "fully_adaptive_fisher", "l2", or "kl"
        binary_reward_fn: optional callable(prompt, response) -> float, tracked
            per step alongside the optimised (float) reward (MBPP+)
        n_chunks: Phase 1 position chunking of the gradient computation
            (doubles automatically on OOM)
        gen_chunks: Phase 1 sampling rollout chunking (increases on OOM)
        track_diagnostics: record Fisher / KL-to-base values per step

    Returns:
        dict with keys: best_response, best_reward, mean_per_iter,
                        best_per_iter, final_reward
    """
    FISHER_REGULARIZERS = ("frozen_fisher", "steered_traj_fisher",
                           "fully_adaptive_fisher")
    assert regularizer in (*FISHER_REGULARIZERS, "l2", "kl"), f"Unknown regularizer: {regularizer}"
    hidden_dim = llm.hidden_dim
    device = prompt_ids.device
    lm_head = llm._lm_head
    lm_weight = lm_head.weight  # (vocab_size, hidden_dim)
    model_dtype = lm_weight.dtype
    eos_id = llm.tokenizer.eos_token_id

    # Cache prompt text for decode (avoid re-decoding every sample)
    prompt_decoded = llm.decode(prompt_ids[0])

    # Base distributions: lazy-initialized from Phase 1 step-0 rollouts (which
    # are at u=0 → base policy). The Fisher gradient is `λ·F·(W·u)`, which is
    # 0 at u=0, so the regularizer is a no-op at step 0 anyway — the fold-in
    # is mathematically exact.
    base_prelogits_per_traj = None # (n_samples, T_base, d) — frozen_fisher
    base_lengths_per_traj = None   # (n_samples,) per-trajectory effective lengths
    T_base = 0
    # Frozen step-0 (u=0) per-trajectory base for the uniform deviation KL —
    # captured for ALL regularizers so every run reports the same quantity.
    base_kl_prelogits_gpu = None   # (n_samples, T_gen, d) on GPU
    base_kl_valid_gpu = None        # (n_samples, T_gen) valid-position mask

    # Pre-compute prompt KV cache for reuse across all sampling calls
    prompt_kv_cache = llm.compute_prompt_kv_cache(prompt_ids, attention_mask)

    # --- Phase 1: Gradient ascent ---
    logger.info(f"  MISVO Phase 1: {n_steps} steps x {n_samples} samples...")
    u = torch.nn.Parameter(torch.zeros(max_new_tokens, hidden_dim, device=device,
                                       dtype=torch.float32))
    optimizer = torch.optim.Adam([u], lr=lr)

    r_best = float("-inf")
    y_best = ""
    r_best_binary = float("-inf")  # best binary reward seen across all steps
    y_best_binary = ""
    mean_rewards = []
    best_rewards = []
    binary_mean_per_iter = [] if binary_reward_fn is not None else None
    binary_best_per_iter = [] if binary_reward_fn is not None else None
    # binary reward of the FLOAT-best sample per iter (couples the two metrics so
    # a "float-selected pass@1 best-so-far" curve — matching the reported reward —
    # can be built; distinct from binary_best_per_iter = oracle max binary).
    binary_at_float_best_per_iter = [] if binary_reward_fn is not None else None
    fisher_per_iter = []
    fully_adaptive_fisher_per_iter = []
    kl_per_iter = []
    l2_per_iter = []
    u_norm_per_iter = []  # ||u|| after each step
    # Per-step generations: list of lists, each inner list has {response, reward}
    generations_per_step = []

    # Timing accumulators
    t_generate = 0.0
    t_score = 0.0
    t_gradient = 0.0
    log_interval = max(1, n_steps // 4)

    # gen_chunks: starting value from caller (default 1); auto-increases on
    # OOM to chunk generation into smaller batches.
    gen_chunks = max(1, int(gen_chunks))

    # GPU-staged base distributions — populated lazily at step 0 below.
    # frozen_fisher: (n_samples, T_base, d).
    base_prelogits_per_traj_gpu = None
    base_valid_mask_gpu = None

    step_bar = tqdm(range(n_steps), desc="  MISVO steps", leave=False)
    for step_k in step_bar:
        # --- Sample N trajectories via HF generate() with hook ---
        t0 = time.time()
        u_detached = u.detach()

        # Generate in chunks with OOM retry (keeps total n_samples)
        while True:
            chunk_size = (n_samples + gen_chunks - 1) // gen_chunks
            all_batch_ids = []
            all_batch_prelogits = []
            oom = False
            for c_start in range(0, n_samples, chunk_size):
                c_n = min(chunk_size, n_samples - c_start)
                try:
                    c_ids, c_prelogs = _safe_batched_sampling(
                        llm, prompt_ids, attention_mask, u_detached,
                        c_n, max_new_tokens, temperature, top_p,
                        prompt_kv_cache=prompt_kv_cache,
                    )
                    all_batch_ids.append(c_ids)
                    all_batch_prelogits.append(c_prelogs)
                except torch.cuda.OutOfMemoryError:
                    del all_batch_ids, all_batch_prelogits
                    torch.cuda.empty_cache()
                    gen_chunks += 1
                    if gen_chunks > n_samples:
                        raise RuntimeError(
                            f"OOM even with chunk_size=1. "
                            f"Reduce n_samples or max_new_tokens.")
                    logger.warning(f"  OOM at step {step_k}, retrying with "
                                   f"{gen_chunks} gen chunks (size={n_samples // gen_chunks})")
                    oom = True
                    break
            if not oom:
                break

        # Concatenate chunks (pad to same seq length if needed)
        max_len = max(o.shape[1] for o in all_batch_ids)
        padded_ids = []
        for o in all_batch_ids:
            if o.shape[1] < max_len:
                pad = torch.full((o.shape[0], max_len - o.shape[1]),
                                 llm.tokenizer.pad_token_id,
                                 dtype=o.dtype, device=o.device)
                o = torch.cat([o, pad], dim=1)
            padded_ids.append(o)
        batch_ids = torch.cat(padded_ids, dim=0)

        max_T = max(p.shape[1] for p in all_batch_prelogits)
        padded_prelogs = []
        for p in all_batch_prelogits:
            if p.shape[1] < max_T:
                pad = torch.zeros(p.shape[0], max_T - p.shape[1], p.shape[2],
                                  dtype=p.dtype)
                p = torch.cat([p, pad], dim=1)
            padded_prelogs.append(p)
        batch_prelogits = torch.cat(padded_prelogs, dim=0)
        # batch_ids: (N, prompt_len + T_gen)
        # batch_prelogits: (N, T_gen, d) on CPU

        T_gen = batch_prelogits.shape[1]

        # Extract generated tokens and compute lengths
        batch_tokens = batch_ids[:, prompt_ids.shape[1]:].cpu()  # (N, T_gen)

        # Compute per-sample generation lengths (position of first EOS + 1, or T_gen)
        eos_mask = (batch_tokens == eos_id)
        has_eos = eos_mask.any(dim=1)
        eos_pos = eos_mask.float().argmax(dim=1)  # first EOS position per sample
        batch_lengths = torch.where(has_eos, eos_pos + 1,
                                    torch.tensor(T_gen, dtype=torch.long))
        lengths_tensor = batch_lengths  # (N,) on CPU

        # Lazy Fisher init: at step 0, u=0 → the rollouts ARE base-policy
        # rollouts, so we can use them directly to estimate base distributions.
        if step_k == 0:
            if regularizer == "frozen_fisher" and base_prelogits_per_traj is None:
                base_prelogits_per_traj = batch_prelogits  # (n_samples, T_gen, d) CPU
                base_lengths_per_traj = batch_lengths.clone()
                T_base = batch_prelogits.shape[1]
                base_prelogits_per_traj_gpu = base_prelogits_per_traj.to(
                    device, dtype=model_dtype, non_blocking=True)
                positions_b = torch.arange(T_base, device=device)
                base_valid_mask_gpu = (
                    positions_b.unsqueeze(0)
                    < base_lengths_per_traj.to(device).unsqueeze(1)
                ).float()
                logger.info(f"  MISVO: bootstrapped per-trajectory base pre-logits "
                            f"from step-0 rollouts (n={n_samples}, T_base={T_base})")
            # Frozen step-0 per-trajectory base for the UNIFORM deviation KL
            # (all regularizers). u=0 here → these are base-policy rollouts.
            if base_kl_prelogits_gpu is None:
                base_kl_prelogits_gpu = batch_prelogits.to(device, dtype=model_dtype)
                _pos_kl = torch.arange(batch_prelogits.shape[1], device=device)
                base_kl_valid_gpu = (
                    _pos_kl.unsqueeze(0) < batch_lengths.to(device).unsqueeze(1)
                ).float()

        # Decode all responses (batched is much faster than per-sample)
        full_texts = llm.batch_decode(batch_ids)
        sample_responses = [t[len(prompt_decoded):] for t in full_texts]
        t_generate += time.time() - t0

        # Score (batched if available)
        t0 = time.time()
        if reward_fn_batch is not None:
            sample_rewards = reward_fn_batch(prompt_text, sample_responses)
        else:
            sample_rewards = []
            for i in tqdm(range(n_samples), desc="    scoring", leave=False):
                sample_rewards.append(reward_fn(prompt_text, sample_responses[i]))
        t_score += time.time() - t0

        # Track best and store
        step_generations = []
        for i in range(n_samples):
            step_generations.append({
                "response": sample_responses[i],
                "reward": sample_rewards[i],
            })
            if sample_rewards[i] > r_best:
                r_best = sample_rewards[i]
                y_best = sample_responses[i]
        generations_per_step.append(step_generations)

        rewards_tensor = torch.tensor(sample_rewards, dtype=torch.float32)

        # Track raw rewards BEFORE any modification (for convergence plots)
        mean_rewards.append(rewards_tensor.mean().item())
        best_at_k = rewards_tensor.max().item()
        best_rewards.append(best_at_k)

        # Binary reward tracking (oracle binary best per step) — runs once,
        # outside the OOM-retry loop below.
        if binary_reward_fn is not None:
            binary_scores = [binary_reward_fn(prompt_text, r) for r in sample_responses]
            binary_mean_per_iter.append(float(np.mean(binary_scores)))
            binary_best_per_iter.append(max(binary_scores))
            binary_at_float_best_per_iter.append(binary_scores[int(np.argmax(sample_rewards))])
            best_bin_idx = int(np.argmax(binary_scores))
            if binary_scores[best_bin_idx] > r_best_binary:
                r_best_binary = binary_scores[best_bin_idx]
                y_best_binary = sample_responses[best_bin_idx]

        # Snapshot state used by the n_chunks-dependent block so we can retry
        # cleanly on OOM (KL modifies rewards_tensor; running normalizer mutates
        # its EMA stats on each call).
        rewards_orig = rewards_tensor.clone()
        norm_snapshot = None
        if reward_normalizer is not None and getattr(reward_normalizer, "mode", "none") == "running":
            norm_snapshot = (reward_normalizer.running_mean,
                             reward_normalizer.running_var,
                             reward_normalizer._initialized)
        n = n_samples
        T = T_gen
        # Local n_chunks for this step — doubles on OOM and is propagated back
        # to the outer scope so subsequent steps start from the new value.
        local_n_chunks = n_chunks
        # Cast u once per step for the chunk slices below (u.data only changes
        # in optimizer.step(), which runs after both blocks below).
        u_data_md = u.data.to(model_dtype)

        # n_chunks-dependent block: KL + advantages + gradient. Auto-retries
        # on OOM by doubling n_chunks (smaller per-chunk allocations).
        batch_prelogits_gpu = batch_tokens_gpu = lengths_gpu = None
        valid_mask_full = valid_f_full = n_valid_per_t = None
        kl_penalties_gpu = kl_penalties = v_full = None
        grad_u = advantages_gpu = advantages = None
        while True:
            try:
                # Reset per-attempt state
                rewards_tensor = rewards_orig.clone()
                if norm_snapshot is not None:
                    (reward_normalizer.running_mean,
                     reward_normalizer.running_var,
                     reward_normalizer._initialized) = norm_snapshot

                # --- Move per-step Phase-1 data to GPU once, build masks (reused by
                # KL regulariser block + gradient block; CPU→GPU per-position slicing
                # was the dominant cost in the legacy implementation) ---
                batch_prelogits_gpu = batch_prelogits.to(device, dtype=model_dtype,
                                                         non_blocking=True)  # (n, T, d)
                batch_tokens_gpu = batch_tokens.to(device, non_blocking=True)  # (n, T)
                lengths_gpu = lengths_tensor.to(device)  # (n,)
                positions = torch.arange(T, device=device)
                valid_mask_full = positions.unsqueeze(0) < lengths_gpu.unsqueeze(1)  # (n, T)
                valid_f_full = valid_mask_full.float()  # (n, T)
                n_valid_per_t = valid_f_full.sum(dim=0).clamp(min=1.0)  # (T,)
                T_eff = int(lengths_gpu.max().item()) if n > 0 else 0
                T_chunk = max(1, (T + local_n_chunks - 1) // local_n_chunks)

                # KL regulariser: modify rewards before computing advantages.
                # R_kl^i = R^i - λ Σ_t [log p_u(y_t^i) - log p_0(y_t^i)]
                # Vectorised: chunk over T, two batched lm_head calls per chunk.
                if regularizer == "kl":
                    with torch.no_grad():
                        kl_penalties_gpu = torch.zeros(n, device=device, dtype=torch.float32)
                        for s in range(0, T_eff, T_chunk):
                            e = min(s + T_chunk, T_eff)
                            tok_chunk = batch_tokens_gpu[:, s:e]
                            valid_chunk = valid_f_full[:, s:e]
                            u_chunk = u_data_md[s:e]
                            z_chunk = batch_prelogits_gpu[:, s:e, :]
                            logits_u = lm_head(z_chunk + u_chunk.unsqueeze(0)).float()
                            logits_0 = lm_head(z_chunk).float()
                            lp_u = F.log_softmax(logits_u, dim=-1).gather(
                                -1, tok_chunk.unsqueeze(-1)).squeeze(-1)
                            lp_0 = F.log_softmax(logits_0, dim=-1).gather(
                                -1, tok_chunk.unsqueeze(-1)).squeeze(-1)
                            kl_penalties_gpu += ((lp_u - lp_0) * valid_chunk).sum(dim=1)
                            del logits_u, logits_0
                    kl_penalties = kl_penalties_gpu.cpu()
                    rewards_tensor = rewards_tensor - lam * kl_penalties

                if reward_normalizer is not None:
                    advantages = reward_normalizer.normalize(rewards_tensor)
                else:
                    advantages = rewards_tensor - rewards_tensor.mean()  # (n_samples,)
                break
            except torch.cuda.OutOfMemoryError:
                # Drop refs to all GPU tensors so empty_cache can free them
                batch_prelogits_gpu = batch_tokens_gpu = lengths_gpu = None
                valid_mask_full = valid_f_full = n_valid_per_t = None
                kl_penalties_gpu = kl_penalties = v_full = None
                grad_u = advantages_gpu = advantages = None
                torch.cuda.empty_cache()
                new_n_chunks = max(local_n_chunks * 2, local_n_chunks + 1)
                if new_n_chunks > T:
                    raise RuntimeError(
                        f"OOM in MISVO KL block even at n_chunks={new_n_chunks} (T={T}). "
                        f"Reduce n_samples or shorten max_new_tokens.")
                logger.warning(f"  OOM in MISVO KL/advantages at step {step_k}, "
                               f"retrying with n_chunks={new_n_chunks} (T_chunk={max(1, (T + new_n_chunks - 1) // new_n_chunks)})")
                local_n_chunks = new_n_chunks
        n_chunks = local_n_chunks  # propagate increase to subsequent steps

        # --- Compute analytical gradient (vectorised, GPU-resident) ---
        # Per chunk: 1 batched lm_head for p_u, plus 1 more for steered_traj_fisher
        # (or 0 for the others). REINFORCE term1 is an embedding lookup
        # (no (n, V) one-hot scatter); term2 collapses i first then does a
        # single (c, V)@W matmul. Fisher direction is computed in V-space and
        # projected to d-space with one (c, V)@W matmul per chunk.
        t0 = time.time()
        # Gradient block retries on OOM by doubling local_n_chunks (smaller
        # T_chunk → smaller per-chunk allocations).
        while True:
            try:
                T_chunk = max(1, (T + local_n_chunks - 1) // local_n_chunks)
                grad_u = torch.zeros(max_new_tokens, hidden_dim, device=device,
                                     dtype=torch.float32)
                # 0-d GPU accumulators — avoid per-chunk .item() syncs.
                step_fisher_value_t = torch.zeros((), device=device, dtype=torch.float32)
                step_fully_adaptive_fisher_value_t = torch.zeros((), device=device, dtype=torch.float32)

                advantages_gpu = advantages.to(device).float()  # (n,)

                # v_t = W u_t for all t in one batched call (used by Fisher regs + FA tracking)
                if regularizer in FISHER_REGULARIZERS:
                    with torch.no_grad():
                        v_full = lm_head(u_data_md).float()  # (T, V)
                else:
                    v_full = None

                for s in range(0, T_eff, T_chunk):
                    e = min(s + T_chunk, T_eff)
                    valid_chunk = valid_f_full[:, s:e]            # (n, c)
                    n_valid_chunk = n_valid_per_t[s:e]            # (c,)
                    adv_per_pos = advantages_gpu.unsqueeze(1) * valid_chunk  # (n, c)

                    with torch.no_grad():
                        # 1. p_u for chunk
                        u_chunk = u_data_md[s:e]                                # (c, d) model_dtype
                        z_chunk = batch_prelogits_gpu[:, s:e, :]               # (n, c, d)
                        logits_u = lm_head(z_chunk + u_chunk.unsqueeze(0))     # (n, c, V)
                        p_u = F.softmax(logits_u.float(), dim=-1)              # (n, c, V) fp32
                        tok_chunk = batch_tokens_gpu[:, s:e]                   # (n, c)
                        del logits_u

                        # 2. REINFORCE direction in d-space
                        W_yt = lm_weight[tok_chunk]                            # (n, c, d) fancy index
                        term1 = (adv_per_pos.unsqueeze(-1) *
                                 W_yt.float()).sum(dim=0) / n                  # (c, d)
                        del W_yt
                        # einsum collapses (n, c, V) reduction without
                        # materialising the n*c*V intermediate (~128MB at V=128k).
                        weighted_p_u = torch.einsum('nc,ncv->cv', adv_per_pos, p_u) / n  # (c, V)
                        term2 = (weighted_p_u.to(model_dtype) @ lm_weight).float()  # (c, d)
                        del weighted_p_u
                        grad_reinforce = term1 - term2                          # (c, d)
                        del term1, term2

                        # 3. Fully-adaptive Fisher *value* — needed for FA gradient
                        #    (pv_fa/pdotv_fa reused below) and as a tracking signal
                        #    for any Fisher regulariser when track_diagnostics=True.
                        pv_fa = pdotv_fa = None
                        need_fa_block = (regularizer == "fully_adaptive_fisher" or
                                         (track_diagnostics and regularizer in FISHER_REGULARIZERS))
                        if need_fa_block:
                            v_chunk = v_full[s:e]                              # (c, V)
                            pv_fa = p_u * v_chunk.unsqueeze(0)                 # (n, c, V)
                            pdotv_fa = pv_fa.sum(dim=-1)                       # (n, c)
                            if track_diagnostics:
                                pv2_sum_fa = (p_u * v_chunk.unsqueeze(0) ** 2).sum(dim=-1)
                                per_traj_var_fa = pv2_sum_fa - pdotv_fa ** 2   # (n, c)
                                mean_per_t_fa = (per_traj_var_fa * valid_chunk
                                                 ).sum(dim=0) / n_valid_chunk  # (c,)
                                step_fully_adaptive_fisher_value_t += 0.5 * mean_per_t_fa.sum()
                                del pv2_sum_fa, per_traj_var_fa

                        # 4. Regulariser-specific gradient term
                        if regularizer == "frozen_fisher":
                            # Average of per-trajectory Fishers (not the Fisher of the averaged distribution).
                            # One extra lm_head pass per chunk on (n_samples, c, d)
                            # → (n_samples, c, V), softmax, then per-traj fisher
                            # gradient averaged across valid trajectories.
                            e_fish = min(e, T_base)
                            if e_fish > s:
                                c_f = e_fish - s
                                z_b = base_prelogits_per_traj_gpu[:, s:e_fish, :]   # (n_samples, c_f, d)
                                v_f = v_full[s:e_fish]                              # (c_f, V) fp32
                                base_logits = lm_head(z_b)                          # (n_samples, c_f, V)
                                p_b_traj = F.softmax(base_logits.float(), dim=-1)   # (n_samples, c_f, V) fp32
                                del base_logits
                                valid_b = base_valid_mask_gpu[:, s:e_fish]          # (n_samples, c_f)
                                n_valid_b = valid_b.sum(dim=0).clamp(min=1.0)       # (c_f,)
                                # Per-traj pv = p_{t,i} * v_t, pdotv per trajectory.
                                pv = p_b_traj * v_f.unsqueeze(0)                    # (n_samples, c_f, V)
                                pdotv = pv.sum(dim=-1)                              # (n_samples, c_f)
                                if track_diagnostics:
                                    pv2_sum = (p_b_traj * v_f.unsqueeze(0) ** 2
                                               ).sum(dim=-1)                       # (n_samples, c_f)
                                    per_traj_var = pv2_sum - pdotv ** 2             # (n_samples, c_f)
                                    mean_per_t = (per_traj_var * valid_b
                                                  ).sum(dim=0) / n_valid_b          # (c_f,)
                                    step_fisher_value_t += 0.5 * mean_per_t.sum()
                                    del pv2_sum, per_traj_var
                                # Avg fisher_vec_V = mean_i (pv_i - p_{t,i}*pdotv_i)
                                # Split-einsum to avoid materialising (n_samples, c, V).
                                weighted_pv = torch.einsum('ic,icv->cv', valid_b, pv)
                                weighted_pbase = torch.einsum(
                                    'ic,icv->cv', valid_b * pdotv, p_b_traj)
                                avg_fisher_vec_V = (weighted_pv - weighted_pbase
                                                    ) / n_valid_b.unsqueeze(-1)     # (c_f, V)
                                del weighted_pv, weighted_pbase, pv, pdotv, p_b_traj
                                fisher_grad_d = (avg_fisher_vec_V.to(model_dtype)
                                                 @ lm_weight).float()                # (c_f, d)
                                grad_u[s:e_fish] = grad_reinforce[:c_f] - lam * fisher_grad_d
                                if e > e_fish:
                                    grad_u[e_fish:e] = grad_reinforce[c_f:]
                                del avg_fisher_vec_V, fisher_grad_d
                            else:
                                grad_u[s:e] = grad_reinforce
                        elif regularizer == "steered_traj_fisher":
                            base_logits = lm_head(z_chunk)                      # (n, c, V)
                            p_base = F.softmax(base_logits.float(), dim=-1)     # (n, c, V) fp32
                            del base_logits
                            pv = p_base * v_chunk.unsqueeze(0)                  # (n, c, V)
                            pdotv = pv.sum(dim=-1)                              # (n, c)
                            if track_diagnostics:
                                pv2_sum = (p_base * v_chunk.unsqueeze(0) ** 2).sum(dim=-1)
                                per_traj_var = pv2_sum - pdotv ** 2              # (n, c)
                                mean_per_t = (per_traj_var * valid_chunk
                                              ).sum(dim=0) / n_valid_chunk       # (c,)
                                step_fisher_value_t += 0.5 * mean_per_t.sum()
                                del pv2_sum, per_traj_var
                            # Split-einsum: avoids materialising fisher_vecs and
                            # the (n, c, V) multiply intermediate (~256MB peak).
                            # avg[c,v] = (1/n_valid[c]) Σ_n valid[n,c] * (pv - p_base*pdotv)
                            weighted_pv = torch.einsum(
                                'nc,ncv->cv', valid_chunk, pv)                   # (c, V)
                            weighted_pbase = torch.einsum(
                                'nc,ncv->cv', valid_chunk * pdotv, p_base)       # (c, V)
                            avg_fisher_vec_V = (
                                weighted_pv - weighted_pbase
                            ) / n_valid_chunk.unsqueeze(-1)                      # (c, V)
                            del weighted_pv, weighted_pbase, pv, pdotv, p_base
                            fisher_grad_d = (avg_fisher_vec_V.to(model_dtype)
                                             @ lm_weight).float()                # (c, d)
                            grad_u[s:e] = grad_reinforce - lam * fisher_grad_d
                        elif regularizer == "fully_adaptive_fisher":
                            # Reuse pv_fa / pdotv_fa from the FA tracking block.
                            # Split-einsum (same trick as steered_traj_fisher).
                            weighted_pv_fa = torch.einsum(
                                'nc,ncv->cv', valid_chunk, pv_fa)                # (c, V)
                            weighted_pu_fa = torch.einsum(
                                'nc,ncv->cv', valid_chunk * pdotv_fa, p_u)       # (c, V)
                            avg_fisher_vec_V = (
                                weighted_pv_fa - weighted_pu_fa
                            ) / n_valid_chunk.unsqueeze(-1)                      # (c, V)
                            del weighted_pv_fa, weighted_pu_fa
                            fisher_grad_d = (avg_fisher_vec_V.to(model_dtype)
                                             @ lm_weight).float()
                            grad_u[s:e] = grad_reinforce - lam * fisher_grad_d
                        elif regularizer == "l2":
                            # ∇_u [E[R] - (λ/2) ||u||²] = REINFORCE - λ u_t
                            grad_u[s:e] = grad_reinforce - lam * u_data_md[s:e].float()
                        else:  # kl, none — plain REINFORCE
                            grad_u[s:e] = grad_reinforce

                # Materialise tracking accumulators once (single sync per step).
                step_fisher_value = step_fisher_value_t.item()
                step_fully_adaptive_fisher_value = step_fully_adaptive_fisher_value_t.item()
                # For fully_adaptive_fisher, the regulariser value IS the FA value.
                if regularizer == "fully_adaptive_fisher":
                    step_fisher_value = step_fully_adaptive_fisher_value
                break
            except torch.cuda.OutOfMemoryError:
                # Drop refs to per-attempt tensors; the persistent ones
                # (batch_prelogits_gpu etc.) stay live and are reused.
                grad_u = None
                advantages_gpu = None
                v_full = None
                torch.cuda.empty_cache()
                new_n_chunks = max(local_n_chunks * 2, local_n_chunks + 1)
                if new_n_chunks > T:
                    raise RuntimeError(
                        f"OOM in MISVO gradient block even at n_chunks={new_n_chunks} (T={T}). "
                        f"Reduce n_samples or shorten max_new_tokens.")
                logger.warning(f"  OOM in MISVO gradient at step {step_k}, "
                               f"retrying with n_chunks={new_n_chunks} (T_chunk={max(1, (T + new_n_chunks - 1) // new_n_chunks)})")
                local_n_chunks = new_n_chunks
        n_chunks = local_n_chunks  # propagate increase to subsequent steps

        # Apply update (Adam minimises, so negate the ascent direction).
        u.grad = -grad_u
        optimizer.step()
        optimizer.zero_grad()
        del batch_prelogits_gpu, batch_tokens_gpu, valid_mask_full, valid_f_full
        if v_full is not None:
            del v_full
        t_gradient += time.time() - t0

        # --- Capture regularizer values ---
        l2_per_iter.append(0.5 * (u.data ** 2).sum().item())
        u_norm_per_iter.append(u.data.norm().item())

        # Fisher diagnostics (regularizer-specific).
        if regularizer in FISHER_REGULARIZERS and track_diagnostics:
            fisher_per_iter.append(step_fisher_value)
            fully_adaptive_fisher_per_iter.append(step_fully_adaptive_fisher_value)
        else:
            fisher_per_iter.append(None)
            fully_adaptive_fisher_per_iter.append(None)
        # UNIFORM deviation KL — identical estimator for ALL regularizers:
        # full-vocab, per-token, summed over positions, averaged over the frozen
        # step-0 (u=0) rollouts, using the APPLIED steering vector u. (The `kl`
        # regularizer still uses `kl_penalties` for its objective above; here we
        # only *report* the same comparable quantity as everyone else.)
        if track_diagnostics and base_kl_prelogits_gpu is not None:
            kl_per_iter.append(kl_deviation(
                base_kl_prelogits_gpu, u.data,
                base_kl_valid_gpu, lm_head, device, model_dtype))
        else:
            kl_per_iter.append(None)

        step_bar.set_postfix(mean=f"{mean_rewards[-1]:.3f}",
                             best_k=f"{best_at_k:.3f}",
                             best=f"{r_best:.3f}")

        if (step_k + 1) % log_interval == 0 or step_k == n_steps - 1:
            total = t_generate + t_score + t_gradient
            logger.info(f"  MISVO timing (step {step_k+1}/{n_steps}): "
                        f"generate={t_generate:.1f}s ({t_generate/total*100:.0f}%) | "
                        f"score={t_score:.1f}s ({t_score/total*100:.0f}%) | "
                        f"gradient={t_gradient:.1f}s ({t_gradient/total*100:.0f}%) | "
                        f"total={total:.1f}s")

    # --- Phase 2: Final greedy generation with optimized u ---
    logger.info("  MISVO Phase 2: Final greedy generation...")
    t0 = time.time()
    final_gen_ids, _ = llm.generate_with_perturbation(
        prompt_ids, attention_mask, u.detach(), max_new_tokens
    )
    full_text_final = llm.decode(final_gen_ids[0])
    response_final = full_text_final[len(prompt_decoded):]
    t_final_gen = time.time() - t0
    t0 = time.time()
    r_final = reward_fn(prompt_text, response_final)
    t_final_score = time.time() - t0

    if r_final > r_best:
        r_best = r_final
        y_best = response_final

    total_all = t_generate + t_score + t_gradient + t_final_gen + t_final_score
    logger.info(f"  MISVO total: generate={t_generate + t_final_gen:.1f}s | "
                f"score={t_score + t_final_score:.1f}s | "
                f"gradient={t_gradient:.1f}s | total={total_all:.1f}s")

    result = {
        "best_response": y_best,
        "best_reward": r_best,
        "mean_per_iter": mean_rewards,
        "best_per_iter": best_rewards,
        "final_reward": r_final,
        "final_response": response_final,
        "steering_vector": u.detach().cpu(),
        "fisher_per_iter": fisher_per_iter,
        "fully_adaptive_fisher_per_iter": fully_adaptive_fisher_per_iter,
        "kl_per_iter": kl_per_iter,
        "l2_per_iter": l2_per_iter,
        "u_norm_per_iter": u_norm_per_iter,
        "generations_per_step": generations_per_step,
        # Phase-1 wall-clock breakdown (seconds), including the Phase-2 final
        # greedy pass. Additive keys for profiling/benchmarks; ignored by the
        # main runner, which selects result keys explicitly.
        "timing": {
            "generate_s": t_generate + t_final_gen,
            "score_s": t_score + t_final_score,
            "gradient_s": t_gradient,
            "total_s": total_all,
        },
    }
    if binary_reward_fn is not None:
        result["best_binary_reward"] = r_best_binary
        result["best_binary_response"] = y_best_binary
        result["binary_mean_per_iter"] = binary_mean_per_iter
        result["binary_best_per_iter"] = binary_best_per_iter
        result["binary_at_float_best_per_iter"] = binary_at_float_best_per_iter
    return result


# ---------------------------------------------------------------------------
# Wave (multi-prompt) MISVO entry point
# ---------------------------------------------------------------------------


def run_misvo_wave(
    llm,
    prompt_ids_list: list,        # list of P (1, L_p) tensors
    attention_mask_list: list,    # list of P (1, L_p)
    prompt_text_list: list,       # list of P strings
    reward_fn_batch_multi,        # (prompts: list, responses: list) -> list[float]
    lam: float = 1.0,
    n_samples: int = 16,
    n_steps: int = 20,
    lr: float = 0.01,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
    reward_normalizer_list=None,  # optional list of P RewardNormalizer
    regularizer: str = "frozen_fisher",
    binary_reward_fn=None,
    n_chunks: int = 4,
    track_diagnostics: bool = True,
) -> list:
    """Run MISVO on P prompts in parallel by pooling generation across the wave.

    Each prompt keeps its own u, Adam optimizer, and history. Generation of
    P*n_samples samples per step runs in one batched call; reward scoring
    flattens across the wave. Gradient computation runs per-prompt over slices
    of the captured prelogits (we do not materialise (P*n, T, V) tensors).

    Returns a list of P result dicts in the same shape as run_misvo.
    """
    FISHER_REGULARIZERS = ("frozen_fisher", "steered_traj_fisher",
                           "fully_adaptive_fisher")
    assert regularizer in (*FISHER_REGULARIZERS, "l2", "kl"), (
        f"Unknown regularizer: {regularizer}")

    P = len(prompt_ids_list)
    hidden_dim = llm.hidden_dim
    device = prompt_ids_list[0].device
    lm_head = llm._lm_head
    lm_weight = lm_head.weight
    model_dtype = lm_weight.dtype
    eos_id = llm.tokenizer.eos_token_id

    # Pad prompts and pre-compute one wave KV cache (re-used across all n_steps).
    padded_ids, padded_mask, _ = llm.pad_and_stack_prompts(
        prompt_ids_list, attention_mask_list)
    prompt_decodes = [llm.decode(ids[0]) for ids in prompt_ids_list]
    prompt_kv_cache = llm.compute_prompt_kv_cache(padded_ids, padded_mask)

    # Base distributions for frozen_fisher are lazy-initialized at step
    # 0 of Phase 1 from the rollouts themselves (which are at u_p=0 → base
    # policy; Fisher gradient is 0 at u=0 so the fold-in is exact). Per-prompt
    # slots start as None.
    base_prelogits_per_traj_list = [None] * P  # (n_samples, T_base, d) — frozen_fisher
    base_lengths_per_traj_list = [None] * P    # (n_samples,) per-prompt lengths
    T_base_list = [0] * P

    # Per-prompt (T, d) u + Adam optimizer.
    u_params = []
    optimizers = []
    for p in range(P):
        up = torch.nn.Parameter(torch.zeros(max_new_tokens, hidden_dim,
                                            device=device, dtype=torch.float32))
        u_params.append(up)
        optimizers.append(torch.optim.Adam([up], lr=lr))

    # Per-prompt GPU-staged base distributions — populated lazily at step 0.
    base_prelogits_per_traj_gpu_list = [None] * P
    base_valid_mask_gpu_list = [None] * P
    # Frozen step-0 (u=0) per-trajectory base for the uniform deviation KL, per
    # prompt — captured for ALL regularizers so every run reports the same thing.
    base_kl_prelogits_gpu_list = [None] * P
    base_kl_valid_gpu_list = [None] * P

    # Per-prompt history.
    r_best = [float("-inf")] * P
    y_best = [""] * P
    r_best_binary = [float("-inf")] * P
    y_best_binary = [""] * P
    mean_per_iter = [[] for _ in range(P)]
    best_per_iter = [[] for _ in range(P)]
    binary_mean_per_iter = [[] for _ in range(P)] if binary_reward_fn is not None else None
    binary_best_per_iter = [[] for _ in range(P)] if binary_reward_fn is not None else None
    binary_at_float_best_per_iter = [[] for _ in range(P)] if binary_reward_fn is not None else None
    fisher_per_iter = [[] for _ in range(P)]
    fully_adaptive_fisher_per_iter = [[] for _ in range(P)]
    kl_per_iter = [[] for _ in range(P)]
    l2_per_iter = [[] for _ in range(P)]
    u_norm_per_iter = [[] for _ in range(P)]  # ||u_p|| after each step
    generations_per_step = [[] for _ in range(P)]

    step_bar = tqdm(range(n_steps), desc="  MISVO wave steps", leave=False)
    for step_k in step_bar:
        # --- Stack per-prompt u into wave perturbations (P, T, d) ---
        u_stack = torch.stack([up.detach() for up in u_params], dim=0)  # (P, T, d)

        # --- Wave generation: P*n_samples sampled trajectories ---
        batch_ids, batch_prelogits = llm.generate_with_perturbation_sampling_wave(
            padded_ids, padded_mask, u_stack, n_per_prompt=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            prompt_kv_cache=prompt_kv_cache,
        )
        # batch_ids: (P*n, max_L+T_gen), batch_prelogits: (P*n, T_gen, d) on CPU.
        T_gen = batch_prelogits.shape[1]
        batch_tokens = batch_ids[:, padded_ids.shape[1]:].cpu()  # (P*n, T_gen)

        # Per-row generation lengths.
        eos_mask = (batch_tokens == eos_id)
        has_eos = eos_mask.any(dim=1)
        eos_pos = eos_mask.float().argmax(dim=1)
        batch_lengths = torch.where(has_eos, eos_pos + 1,
                                    torch.tensor(T_gen, dtype=torch.long))

        # Decode all and split per prompt.
        full_texts = llm.batch_decode(batch_ids)
        responses_per_prompt = [[] for _ in range(P)]
        for row, ft in enumerate(full_texts):
            p = row // n_samples
            responses_per_prompt[p].append(ft[len(prompt_decodes[p]):])

        # --- Wave reward scoring (one batched RM call across all P*n responses) ---
        flat_prompts, flat_responses = [], []
        for p in range(P):
            for resp in responses_per_prompt[p]:
                flat_prompts.append(prompt_text_list[p])
                flat_responses.append(resp)
        flat_rewards = reward_fn_batch_multi(flat_prompts, flat_responses)

        # --- Per-prompt: gradient + Adam step + tracking ---
        for p in range(P):
            sl = slice(p * n_samples, (p + 1) * n_samples)
            prelogits_p = batch_prelogits[sl]                  # (n, T_gen, d) on CPU
            tokens_p = batch_tokens[sl]                        # (n, T_gen)
            lengths_p = batch_lengths[sl]                       # (n,)

            # Lazy Fisher init for this prompt: at step 0 (u_p=0) the rollouts
            # are base-policy → use them to populate base distributions.
            if step_k == 0:
                if regularizer == "frozen_fisher" \
                        and base_prelogits_per_traj_list[p] is None:
                    base_prelogits_per_traj_list[p] = prelogits_p   # (n, T_gen, d) CPU
                    base_lengths_per_traj_list[p] = lengths_p.clone()
                    T_b = prelogits_p.shape[1]
                    T_base_list[p] = T_b
                    base_prelogits_per_traj_gpu_list[p] = prelogits_p.to(
                        device, dtype=model_dtype, non_blocking=True)
                    positions_b = torch.arange(T_b, device=device)
                    base_valid_mask_gpu_list[p] = (
                        positions_b.unsqueeze(0)
                        < lengths_p.to(device).unsqueeze(1)
                    ).float()
                # Frozen per-trajectory base for the UNIFORM deviation KL (all regs).
                if base_kl_prelogits_gpu_list[p] is None:
                    base_kl_prelogits_gpu_list[p] = prelogits_p.to(device, dtype=model_dtype)
                    _pos_klw = torch.arange(prelogits_p.shape[1], device=device)
                    base_kl_valid_gpu_list[p] = (
                        _pos_klw.unsqueeze(0) < lengths_p.to(device).unsqueeze(1)
                    ).float()

            responses_p = responses_per_prompt[p]
            rewards_p = flat_rewards[p * n_samples:(p + 1) * n_samples]
            rewards_t = torch.tensor(rewards_p, dtype=torch.float32)

            generations_per_step[p].append(
                [{"response": responses_p[i], "reward": rewards_p[i]} for i in range(n_samples)])
            for i in range(n_samples):
                if rewards_p[i] > r_best[p]:
                    r_best[p] = rewards_p[i]
                    y_best[p] = responses_p[i]
            mean_per_iter[p].append(rewards_t.mean().item())
            best_per_iter[p].append(rewards_t.max().item())
            if binary_reward_fn is not None:
                bin_scores = [binary_reward_fn(prompt_text_list[p], r) for r in responses_p]
                binary_mean_per_iter[p].append(float(np.mean(bin_scores)))
                binary_best_per_iter[p].append(max(bin_scores))
                binary_at_float_best_per_iter[p].append(bin_scores[int(np.argmax(rewards_p))])
                bb_idx = int(np.argmax(bin_scores))
                if bin_scores[bb_idx] > r_best_binary[p]:
                    r_best_binary[p] = bin_scores[bb_idx]
                    y_best_binary[p] = responses_p[bb_idx]

            # KL regulariser modifies rewards.
            local_n_chunks = n_chunks
            n = n_samples
            T = T_gen
            T_chunk = max(1, (T + local_n_chunks - 1) // local_n_chunks)
            up = u_params[p]

            prelogits_gpu = prelogits_p.to(device, dtype=model_dtype, non_blocking=True)
            tokens_gpu = tokens_p.to(device, non_blocking=True)
            lengths_gpu = lengths_p.to(device)
            positions = torch.arange(T, device=device)
            valid_mask_full = positions.unsqueeze(0) < lengths_gpu.unsqueeze(1)  # (n, T)
            valid_f_full = valid_mask_full.float()
            n_valid_per_t = valid_f_full.sum(dim=0).clamp(min=1.0)
            T_eff = int(lengths_gpu.max().item())

            u_data_md = up.data.to(model_dtype)
            kl_penalties = None
            if regularizer == "kl":
                with torch.no_grad():
                    kl_pen_gpu = torch.zeros(n, device=device, dtype=torch.float32)
                    for s in range(0, T_eff, T_chunk):
                        e = min(s + T_chunk, T_eff)
                        tok_chunk = tokens_gpu[:, s:e]
                        valid_chunk = valid_f_full[:, s:e]
                        u_chunk = u_data_md[s:e]
                        z_chunk = prelogits_gpu[:, s:e, :]
                        logits_u = lm_head(z_chunk + u_chunk.unsqueeze(0)).float()
                        logits_0 = lm_head(z_chunk).float()
                        lp_u = F.log_softmax(logits_u, dim=-1).gather(
                            -1, tok_chunk.unsqueeze(-1)).squeeze(-1)
                        lp_0 = F.log_softmax(logits_0, dim=-1).gather(
                            -1, tok_chunk.unsqueeze(-1)).squeeze(-1)
                        kl_pen_gpu += ((lp_u - lp_0) * valid_chunk).sum(dim=1)
                        del logits_u, logits_0
                kl_penalties = kl_pen_gpu.cpu()
                rewards_t = rewards_t - lam * kl_penalties

            if reward_normalizer_list is not None and reward_normalizer_list[p] is not None:
                advantages = reward_normalizer_list[p].normalize(rewards_t)
            else:
                advantages = rewards_t - rewards_t.mean()

            # --- Gradient block (mirrors single-prompt path; per-prompt only) ---
            grad_u = torch.zeros(max_new_tokens, hidden_dim, device=device,
                                 dtype=torch.float32)
            step_fisher_value_t = torch.zeros((), device=device, dtype=torch.float32)
            step_fa_value_t = torch.zeros((), device=device, dtype=torch.float32)
            advantages_gpu = advantages.to(device).float()
            v_full = None
            if regularizer in FISHER_REGULARIZERS:
                with torch.no_grad():
                    v_full = lm_head(u_data_md).float()  # (T, V)

            for s in range(0, T_eff, T_chunk):
                e = min(s + T_chunk, T_eff)
                valid_chunk = valid_f_full[:, s:e]
                n_valid_chunk = n_valid_per_t[s:e]
                adv_per_pos = advantages_gpu.unsqueeze(1) * valid_chunk
                with torch.no_grad():
                    u_chunk = u_data_md[s:e]
                    z_chunk = prelogits_gpu[:, s:e, :]
                    logits_u = lm_head(z_chunk + u_chunk.unsqueeze(0))
                    p_u = F.softmax(logits_u.float(), dim=-1)
                    tok_chunk = tokens_gpu[:, s:e]
                    del logits_u

                    W_yt = lm_weight[tok_chunk]
                    term1 = (adv_per_pos.unsqueeze(-1) * W_yt.float()).sum(dim=0) / n
                    del W_yt
                    weighted_p_u = torch.einsum('nc,ncv->cv', adv_per_pos, p_u) / n
                    term2 = (weighted_p_u.to(model_dtype) @ lm_weight).float()
                    del weighted_p_u
                    grad_reinforce = term1 - term2
                    del term1, term2

                    pv_fa = pdotv_fa = None
                    need_fa_block = (regularizer == "fully_adaptive_fisher" or
                                     (track_diagnostics and regularizer in FISHER_REGULARIZERS))
                    if need_fa_block:
                        v_chunk = v_full[s:e]
                        pv_fa = p_u * v_chunk.unsqueeze(0)
                        pdotv_fa = pv_fa.sum(dim=-1)
                        if track_diagnostics:
                            pv2_sum_fa = (p_u * v_chunk.unsqueeze(0) ** 2).sum(dim=-1)
                            per_traj_var_fa = pv2_sum_fa - pdotv_fa ** 2
                            mean_per_t_fa = (per_traj_var_fa * valid_chunk
                                             ).sum(dim=0) / n_valid_chunk
                            step_fa_value_t += 0.5 * mean_per_t_fa.sum()
                            del pv2_sum_fa, per_traj_var_fa

                    if regularizer == "frozen_fisher":
                        # Mean of per-trajectory Fishers (not the Fisher of the averaged distribution).
                        e_fish = min(e, T_base_list[p])
                        if e_fish > s:
                            c_f = e_fish - s
                            z_b = base_prelogits_per_traj_gpu_list[p][:, s:e_fish, :]
                            v_f = v_full[s:e_fish]
                            base_logits = lm_head(z_b)                          # (n_samples, c_f, V)
                            p_b_traj = F.softmax(base_logits.float(), dim=-1)
                            del base_logits
                            valid_b = base_valid_mask_gpu_list[p][:, s:e_fish]
                            n_valid_b = valid_b.sum(dim=0).clamp(min=1.0)
                            pv = p_b_traj * v_f.unsqueeze(0)
                            pdotv = pv.sum(dim=-1)
                            if track_diagnostics:
                                pv2_sum = (p_b_traj * v_f.unsqueeze(0) ** 2
                                           ).sum(dim=-1)
                                per_traj_var = pv2_sum - pdotv ** 2
                                mean_per_t = (per_traj_var * valid_b
                                              ).sum(dim=0) / n_valid_b
                                step_fisher_value_t += 0.5 * mean_per_t.sum()
                                del pv2_sum, per_traj_var
                            weighted_pv = torch.einsum('ic,icv->cv', valid_b, pv)
                            weighted_pbase = torch.einsum(
                                'ic,icv->cv', valid_b * pdotv, p_b_traj)
                            avg_fisher_vec_V = (weighted_pv - weighted_pbase
                                                ) / n_valid_b.unsqueeze(-1)
                            del weighted_pv, weighted_pbase, pv, pdotv, p_b_traj
                            fisher_grad_d = (avg_fisher_vec_V.to(model_dtype)
                                             @ lm_weight).float()
                            grad_u[s:e_fish] = grad_reinforce[:c_f] - lam * fisher_grad_d
                            if e > e_fish:
                                grad_u[e_fish:e] = grad_reinforce[c_f:]
                            del avg_fisher_vec_V, fisher_grad_d
                        else:
                            grad_u[s:e] = grad_reinforce
                    elif regularizer == "steered_traj_fisher":
                        base_logits = lm_head(z_chunk)
                        p_base = F.softmax(base_logits.float(), dim=-1)
                        del base_logits
                        pv = p_base * v_chunk.unsqueeze(0)
                        pdotv = pv.sum(dim=-1)
                        if track_diagnostics:
                            pv2_sum = (p_base * v_chunk.unsqueeze(0) ** 2).sum(dim=-1)
                            per_traj_var = pv2_sum - pdotv ** 2
                            mean_per_t = (per_traj_var * valid_chunk
                                          ).sum(dim=0) / n_valid_chunk
                            step_fisher_value_t += 0.5 * mean_per_t.sum()
                            del pv2_sum, per_traj_var
                        weighted_pv = torch.einsum('nc,ncv->cv', valid_chunk, pv)
                        weighted_pbase = torch.einsum(
                            'nc,ncv->cv', valid_chunk * pdotv, p_base)
                        avg_fisher_vec_V = (weighted_pv - weighted_pbase
                                            ) / n_valid_chunk.unsqueeze(-1)
                        del weighted_pv, weighted_pbase, pv, pdotv, p_base
                        fisher_grad_d = (avg_fisher_vec_V.to(model_dtype)
                                         @ lm_weight).float()
                        grad_u[s:e] = grad_reinforce - lam * fisher_grad_d
                    elif regularizer == "fully_adaptive_fisher":
                        weighted_pv_fa = torch.einsum('nc,ncv->cv', valid_chunk, pv_fa)
                        weighted_pu_fa = torch.einsum(
                            'nc,ncv->cv', valid_chunk * pdotv_fa, p_u)
                        avg_fisher_vec_V = (weighted_pv_fa - weighted_pu_fa
                                            ) / n_valid_chunk.unsqueeze(-1)
                        del weighted_pv_fa, weighted_pu_fa
                        fisher_grad_d = (avg_fisher_vec_V.to(model_dtype)
                                         @ lm_weight).float()
                        grad_u[s:e] = grad_reinforce - lam * fisher_grad_d
                    elif regularizer == "l2":
                        grad_u[s:e] = grad_reinforce - lam * u_data_md[s:e].float()
                    else:  # kl
                        grad_u[s:e] = grad_reinforce

            step_fisher_value = step_fisher_value_t.item()
            step_fa_value = step_fa_value_t.item()
            if regularizer == "fully_adaptive_fisher":
                step_fisher_value = step_fa_value

            up.grad = -grad_u
            optimizers[p].step()
            optimizers[p].zero_grad()

            l2_per_iter[p].append(0.5 * (up.data ** 2).sum().item())
            u_norm_per_iter[p].append(up.data.norm().item())
            # Fisher diagnostics (regularizer-specific).
            if regularizer in FISHER_REGULARIZERS and track_diagnostics:
                fisher_per_iter[p].append(step_fisher_value)
                fully_adaptive_fisher_per_iter[p].append(step_fa_value)
            else:
                fisher_per_iter[p].append(None)
                fully_adaptive_fisher_per_iter[p].append(None)
            # UNIFORM deviation KL — identical estimator for ALL regularizers
            # (full-vocab, per-token, summed, averaged over the frozen step-0
            # per-trajectory base, using the APPLIED steering vector).
            if track_diagnostics and base_kl_prelogits_gpu_list[p] is not None:
                kl_per_iter[p].append(kl_deviation(
                    base_kl_prelogits_gpu_list[p], up.data,
                    base_kl_valid_gpu_list[p], lm_head, device, model_dtype))
            else:
                kl_per_iter[p].append(None)

            del prelogits_gpu, tokens_gpu, valid_mask_full, valid_f_full
            if v_full is not None:
                del v_full

    # --- Phase 2: per-prompt final greedy generation with optimized u ---
    logger.info("  MISVO wave Phase 2: final greedy generation per prompt...")
    final_responses = []
    final_rewards = []
    for p in range(P):
        gen_ids, _ = llm.generate_with_perturbation(
            prompt_ids_list[p], attention_mask_list[p],
            u_params[p].detach(), max_new_tokens)
        full_text = llm.decode(gen_ids[0])
        resp = full_text[len(prompt_decodes[p]):]
        final_responses.append(resp)
    final_rewards = reward_fn_batch_multi(prompt_text_list, final_responses)
    for p in range(P):
        if final_rewards[p] > r_best[p]:
            r_best[p] = final_rewards[p]
            y_best[p] = final_responses[p]

    results = []
    for p in range(P):
        res = {
            "best_response": y_best[p],
            "best_reward": r_best[p],
            "mean_per_iter": mean_per_iter[p],
            "best_per_iter": best_per_iter[p],
            "final_reward": final_rewards[p],
            "final_response": final_responses[p],
            "steering_vector": u_params[p].detach().cpu(),
            "fisher_per_iter": fisher_per_iter[p],
            "fully_adaptive_fisher_per_iter": fully_adaptive_fisher_per_iter[p],
            "kl_per_iter": kl_per_iter[p],
            "l2_per_iter": l2_per_iter[p],
            "u_norm_per_iter": u_norm_per_iter[p],
            "generations_per_step": generations_per_step[p],
        }
        if binary_reward_fn is not None:
            res["best_binary_reward"] = r_best_binary[p]
            res["best_binary_response"] = y_best_binary[p]
            res["binary_mean_per_iter"] = binary_mean_per_iter[p]
            res["binary_best_per_iter"] = binary_best_per_iter[p]
            res["binary_at_float_best_per_iter"] = binary_at_float_best_per_iter[p]
        results.append(res)
    return results
