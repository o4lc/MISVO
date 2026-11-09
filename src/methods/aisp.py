"""
AISP: Adaptive Importance Sampling on Pre-logits
Algorithm 1 from "Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space"
(Kanai et al., ICML 2026, arXiv:2510.26219)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import logger


def kl_deviation(base_prelogits, u, valid_mask, lm_head, device, model_dtype,
                 chunk_size=8):
    """Uniform 'deviation' KL used for reporting across AISP and MISVO.

        KL_dev = (1/n) Σ_i Σ_t KL( softmax(W(z_t^i + u_t)) || softmax(W z_t^i) )

    i.e. the FULL-vocabulary forward KL of the next-token distribution at each
    position, SUMMED over valid token positions, AVERAGED over the n trajectories
    that are valid at that position. `z_t^i` = per-trajectory base (u=0) pre-logits;
    `u` = the APPLIED steering vector (u_hat for AISP, u for MISVO). This is the
    single quantity every method reports so the numbers are comparable.

    Args:
        base_prelogits: (n, T, d) per-trajectory base pre-logits.
        u:              (T, d) applied steering vector.
        valid_mask:     (n, T) in {0,1}; position t valid for trajectory i.
    Returns: float
    """
    n, T0, d = base_prelogits.shape
    T = min(T0, u.shape[0], valid_mask.shape[1])
    total = 0.0
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        z = base_prelogits[:, s:e].to(device, dtype=model_dtype)          # (n,c,d)
        uu = u[s:e].to(device, dtype=model_dtype).unsqueeze(0)            # (1,c,d)
        vm = valid_mask[:, s:e].to(device).float()                       # (n,c)
        with torch.no_grad():
            log_p_base = F.log_softmax(lm_head(z).float(), dim=-1)        # (n,c,V)
            log_p_pert = F.log_softmax(lm_head(z + uu).float(), dim=-1)   # (n,c,V)
            kl_tc = (log_p_pert.exp() * (log_p_pert - log_p_base)).sum(-1)  # (n,c)
            n_valid = vm.sum(0).clamp(min=1.0)                            # (c,)
            total += ((kl_tc * vm).sum(0) / n_valid).sum().item()        # mean over valid traj, sum pos
    return total


class RewardNormalizer:
    """Normalizes rewards to stabilize importance weights / REINFORCE gradients.

    Modes:
        "none": no normalization (raw rewards)
        "batch": per-batch z-score: (r - batch_mean) / (batch_std + eps)
        "running": running z-score with EMA of mean/std across iterations
    """

    def __init__(self, mode: str = "none", momentum: float = 0.1, eps: float = 1e-8):
        self.mode = mode
        self.momentum = momentum
        self.eps = eps
        self.running_mean = 0.0
        self.running_var = 1.0
        self._initialized = False

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of rewards. Returns a new tensor."""
        if self.mode == "none":
            return rewards

        # With a single sample, variance is undefined — skip normalization
        if rewards.numel() <= 1:
            return rewards - rewards.mean()

        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()

        if self.mode == "batch":
            std = (batch_var + self.eps) ** 0.5
            return (rewards - batch_mean) / std

        if self.mode == "running":
            if not self._initialized:
                self.running_mean = batch_mean
                self.running_var = batch_var
                self._initialized = True
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            std = (self.running_var + self.eps) ** 0.5
            return (rewards - self.running_mean) / std

        raise ValueError(f"Unknown normalization mode: {self.mode}")


def run_aisp_with_stored_rewards(
    llm,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reward_fn,
    prompt_text: str,
    lam: float = 0.5,
    alpha: float = 0.999,
    sigma_sq: float = 0.5,
    n: int = 32,
    kappa: int = 32,
    max_new_tokens: int = 128,
    batch_size: int = 32,
    reward_fn_batch=None,
    reward_normalizer: "RewardNormalizer | None" = None,
    compute_regularizers: bool = False,
    binary_reward_fn=None,
) -> dict:
    """Faithful AISP: generates responses AND scores them iteration by iteration.

    This is the primary implementation matching Algorithm 1.
    Reward model must be loaded during this call.

    Args:
        batch_size: generation batch size (n samples generated in parallel)
        reward_fn_batch: optional callable(prompt, [responses]) -> [float] for batched scoring
        reward_normalizer: optional RewardNormalizer for stabilizing importance weights
        compute_regularizers: if True, capture pre-logits and compute Fisher/KL values per iter
    """
    hidden_dim = llm.hidden_dim
    sigma = sigma_sq ** 0.5

    u_hat = torch.zeros(max_new_tokens, hidden_dim)
    r_best = float("-inf")
    y_best = ""
    r_best_binary = float("-inf")
    y_best_binary = ""
    generations_per_step = []
    mean_rewards = []
    best_rewards = []
    binary_mean_per_iter = [] if binary_reward_fn is not None else None
    binary_best_per_iter = [] if binary_reward_fn is not None else None
    # binary of the FLOAT-best sample per iter (see misvo.py) — enables a
    # "float-selected pass@1 best-so-far" curve that matches the reported reward.
    binary_at_float_best_per_iter = [] if binary_reward_fn is not None else None
    fisher_per_iter = []
    token_kl_per_iter = []
    gaussian_kl_per_iter = []
    l2_per_iter = []
    eff_n_per_iter = []
    prompt_text_decoded = llm.decode(prompt_ids[0])

    # Timing accumulators
    t_generate = 0.0
    t_score = 0.0
    t_update = 0.0
    log_interval = max(1, kappa // 4)  # log timing ~4 times during run

    # Pre-compute prompt KV cache for reuse across iterations
    prompt_kv_cache = llm.compute_prompt_kv_cache(prompt_ids, attention_mask)

    # Clean u=0 base-policy rollouts for the UNIFORM deviation KL (generated once,
    # reused every iteration). AISP's iteration rollouts are noise-perturbed
    # (v = u_hat + sigma*noise), so they are NOT a valid base — hence a separate
    # zero-perturbation sample here, matching MISVO's step-0 base rollouts.
    base_kl_prelogits_gpu = None
    base_kl_valid_gpu = None
    if compute_regularizers:
        _dev = next(llm._lm_head.parameters()).device
        _mdt = next(llm._lm_head.parameters()).dtype
        _zero = torch.zeros(n, max_new_tokens, hidden_dim)
        _pre_chunks, _len_chunks = [], []
        _gs = 0
        while _gs < n:
            _cb = min(batch_size, n - _gs)
            _out, _pre = llm.generate_with_perturbation_batch(
                prompt_ids, attention_mask, _zero[_gs:_gs + _cb], max_new_tokens,
                prompt_kv_cache=prompt_kv_cache, capture_prelogits=True)
            _pre_chunks.append(_pre)
            _gen = _out[:, prompt_ids.shape[1]:].cpu()
            _eos = (_gen == llm.tokenizer.eos_token_id)
            _has = _eos.any(dim=1)
            _len = torch.where(_has, _eos.float().argmax(dim=1).long() + 1,
                               torch.full((_gen.shape[0],), _gen.shape[1], dtype=torch.long))
            _len_chunks.append(_len)
            _gs += _cb
        _Tm = max(p.shape[1] for p in _pre_chunks)
        _pad = []
        for p in _pre_chunks:
            if p.shape[1] < _Tm:
                p = torch.cat([p, torch.zeros(p.shape[0], _Tm - p.shape[1],
                                              p.shape[2], dtype=p.dtype)], dim=1)
            _pad.append(p)
        base_kl_prelogits_gpu = torch.cat(_pad, dim=0).to(_dev, dtype=_mdt)  # (n, Tm, d)
        _lens = torch.cat(_len_chunks, dim=0)                                # (n,)
        base_kl_valid_gpu = (torch.arange(_Tm).unsqueeze(0)
                             < _lens.unsqueeze(1)).float().to(_dev)          # (n, Tm)

    iter_bar = tqdm(range(kappa), desc="  AISP iters", leave=False)
    for k in iter_bar:
        responses_k = []
        rewards_k = []

        # Sample all n perturbations (vectorized)
        t0 = time.time()
        v_batch = u_hat.unsqueeze(0) + sigma * torch.randn(n, max_new_tokens, hidden_dim)
        all_prelogits_k = [] if compute_regularizers else None
        gen_start = 0
        while gen_start < n:
            cur_batch = min(batch_size, n - gen_start)
            try:
                gen_result = llm.generate_with_perturbation_batch(
                    prompt_ids, attention_mask,
                    v_batch[gen_start:gen_start + cur_batch], max_new_tokens,
                    prompt_kv_cache=prompt_kv_cache,
                    capture_prelogits=compute_regularizers,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logger.warning(f"  OOM at iter {k}, reducing batch_size to {batch_size}")
                continue  # retry same gen_start with smaller batch
            if compute_regularizers:
                outputs, batch_prelogits = gen_result
                all_prelogits_k.append(batch_prelogits)  # (cur_batch, T_gen, d)
            else:
                outputs = gen_result
            full_texts = llm.batch_decode(outputs)
            for ft in full_texts:
                responses_k.append(ft[len(prompt_text_decoded):])
            gen_start += cur_batch
        t_generate += time.time() - t0

        # Score all n responses (batched if available)
        t0 = time.time()
        if reward_fn_batch is not None:
            rewards_k = reward_fn_batch(prompt_text, responses_k)
        else:
            rewards_k = []
            for i in tqdm(range(n), desc="    scoring", leave=False):
                r = reward_fn(prompt_text, responses_k[i])
                rewards_k.append(r)
        t_score += time.time() - t0

        # Collect per-step generations
        step_generations = [{"response": responses_k[i], "reward": rewards_k[i]}
                            for i in range(n)]
        generations_per_step.append(step_generations)

        rewards_tensor = torch.tensor(rewards_k, dtype=torch.float32)
        mean_rewards.append(rewards_tensor.mean().item())
        best_at_k = rewards_tensor.max().item()
        best_rewards.append(best_at_k)

        best_idx = rewards_tensor.argmax().item()
        if rewards_k[best_idx] > r_best:
            r_best = rewards_k[best_idx]
            y_best = responses_k[best_idx]

        # Binary reward tracking (oracle binary best per step)
        if binary_reward_fn is not None:
            import numpy as np
            binary_scores = [binary_reward_fn(prompt_text, r) for r in responses_k]
            binary_mean_per_iter.append(float(np.mean(binary_scores)))
            binary_best_per_iter.append(max(binary_scores))
            binary_at_float_best_per_iter.append(binary_scores[int(np.argmax(rewards_k))])
            best_bin_idx = int(np.argmax(binary_scores))
            if binary_scores[best_bin_idx] > r_best_binary:
                r_best_binary = binary_scores[best_bin_idx]
                y_best_binary = responses_k[best_bin_idx]

        # Normalize rewards for importance weight computation
        t0 = time.time()
        if reward_normalizer is not None:
            rewards_for_weights = reward_normalizer.normalize(rewards_tensor)
        else:
            rewards_for_weights = rewards_tensor

        # Importance-weighted update (Eq. 18) — vectorized
        # dot_products[i] = sum_t u_hat_t^T v_t^i
        dot_products = (u_hat.unsqueeze(0) * v_batch).sum(dim=(1, 2))  # (n,)
        scores = (1.0 / lam) * rewards_for_weights - ((1.0 - alpha) / sigma_sq) * dot_products

        weights = F.softmax(scores, dim=0)
        u_hat = (weights.unsqueeze(1).unsqueeze(2) * v_batch).sum(dim=0)
        t_update += time.time() - t0

        eff_n = 1.0 / (weights ** 2).sum().item()
        eff_n_per_iter.append(eff_n)

        # Regularizer values
        l2_val = (u_hat ** 2).sum().item()
        l2_per_iter.append(l2_val)
        gaussian_kl_per_iter.append(l2_val / sigma_sq)

        if compute_regularizers and all_prelogits_k:
            # Pad pre-logits to max_new_tokens (batches may differ in T_gen
            # because different samples hit EOS at different points)
            max_T = max(p.shape[1] for p in all_prelogits_k)
            padded = []
            for p in all_prelogits_k:
                if p.shape[1] < max_T:
                    pad = torch.zeros(p.shape[0], max_T - p.shape[1], p.shape[2],
                                      dtype=p.dtype, device=p.device)
                    p = torch.cat([p, pad], dim=1)
                padded.append(p)
            # Compute per-trajectory Fisher and token-level KL
            prelogits_all = torch.cat(padded, dim=0)  # (n, max_T, d)
            T_gen_k = prelogits_all.shape[1]
            device = next(llm._lm_head.parameters()).device
            model_dtype = next(llm._lm_head.parameters()).dtype
            lm_head = llm._lm_head

            fisher_total = 0.0
            kl_total = 0.0
            T_compute = min(T_gen_k, max_new_tokens)
            chunk_size = 8  # process positions in chunks to limit memory

            # Vectorised: each chunk processes c positions at once for all n
            # trajectories. Tensors are shape (n, c, V); 3 lm_head calls per
            # chunk replace 3*c calls in the per-position version.
            for t_start in range(0, T_compute, chunk_size):
                t_end = min(t_start + chunk_size, T_compute)
                z_chunk = prelogits_all[:, t_start:t_end, :].to(device, dtype=model_dtype)  # (n, c, d)
                v_chunk = v_batch[:, t_start:t_end, :].to(device, dtype=model_dtype)        # (n, c, d)

                with torch.no_grad():
                    # Fisher: Var_{p_base^i}(W v_t^i) per trajectory, per position
                    base_logits = lm_head(z_chunk).float()                  # (n, c, V)
                    p_base = F.softmax(base_logits, dim=-1)
                    Wv = lm_head(v_chunk).float()                           # (n, c, V)
                    pWv = (p_base * Wv).sum(dim=-1)                         # (n, c)
                    pWv2 = (p_base * Wv ** 2).sum(dim=-1)                   # (n, c)
                    var_per_traj = pWv2 - pWv ** 2                          # (n, c)
                    # mean over n trajectories, sum over c positions
                    fisher_total += 0.5 * var_per_traj.mean(dim=0).sum().item()

                    del base_logits, p_base, Wv

            fisher_per_iter.append(fisher_total)
            # UNIFORM deviation KL — full-vocab, per-token, summed, averaged over
            # the clean base rollouts, using the APPLIED steering vector u_hat
            # (NOT the noisy per-sample controls v_batch). Same estimator as MISVO.
            if base_kl_prelogits_gpu is not None:
                token_kl_per_iter.append(kl_deviation(
                    base_kl_prelogits_gpu, u_hat, base_kl_valid_gpu,
                    lm_head, device, model_dtype))
            else:
                token_kl_per_iter.append(None)
            del prelogits_all
        else:
            fisher_per_iter.append(None)
            token_kl_per_iter.append(None)

        iter_bar.set_postfix(mean=f"{mean_rewards[-1]:.3f}",
                             best_k=f"{best_at_k:.3f}",
                             best=f"{r_best:.3f}",
                             eff_n=f"{eff_n:.1f}")

        if (k + 1) % log_interval == 0 or k == kappa - 1:
            total = t_generate + t_score + t_update
            logger.info(f"  AISP timing (iter {k+1}/{kappa}): "
                        f"generate={t_generate:.1f}s ({t_generate/total*100:.0f}%) | "
                        f"score={t_score:.1f}s ({t_score/total*100:.0f}%) | "
                        f"update={t_update:.1f}s ({t_update/total*100:.0f}%) | "
                        f"total={total:.1f}s")

    # Final sample from optimized distribution.
    t0 = time.time()
    v_final = u_hat + sigma * torch.randn(max_new_tokens, hidden_dim)
    gen_ids_final = llm.generate_with_perturbation_batch(
        prompt_ids, attention_mask, v_final.unsqueeze(0), max_new_tokens,
        prompt_kv_cache=prompt_kv_cache,
    )
    full_text_final = llm.decode(gen_ids_final[0])
    response_final = full_text_final[len(prompt_text_decoded):]
    t_final_gen = time.time() - t0
    t0 = time.time()
    r_final = reward_fn(prompt_text, response_final)
    t_final_score = time.time() - t0

    if r_final > r_best:
        r_best = r_final
        y_best = response_final

    total_all = t_generate + t_score + t_update + t_final_gen + t_final_score
    logger.info(f"  AISP total: generate={t_generate + t_final_gen:.1f}s | "
                f"score={t_score + t_final_score:.1f}s | "
                f"update={t_update:.1f}s | total={total_all:.1f}s")

    result = {
        "best_response": y_best,
        "best_reward": r_best,
        "mean_per_iter": mean_rewards,
        "best_per_iter": best_rewards,
        "final_reward": r_final,
        "final_response": response_final,
        "steering_vector": u_hat.detach().cpu(),
        "fisher_per_iter": fisher_per_iter,
        "token_kl_per_iter": token_kl_per_iter,
        "gaussian_kl_per_iter": gaussian_kl_per_iter,
        "l2_per_iter": l2_per_iter,
        "eff_n_per_iter": eff_n_per_iter,
        "generations_per_step": generations_per_step,
    }
    if binary_mean_per_iter is not None:
        result["best_binary_reward"] = r_best_binary
        result["best_binary_response"] = y_best_binary
        result["binary_mean_per_iter"] = binary_mean_per_iter
        result["binary_best_per_iter"] = binary_best_per_iter
        result["binary_at_float_best_per_iter"] = binary_at_float_best_per_iter
    return result


# ---------------------------------------------------------------------------
# Wave (multi-prompt) AISP entry point
# ---------------------------------------------------------------------------


def run_aisp_with_stored_rewards_wave(
    llm,
    prompt_ids_list: list,        # list of P (1, L_p) tensors
    attention_mask_list: list,    # list of P (1, L_p)
    prompt_text_list: list,       # list of P prompt strings
    reward_fn_batch_multi,        # (prompts: list, responses: list) -> list[float]
    lam: float = 0.5,
    alpha: float = 0.999,
    sigma_sq: float = 0.5,
    n: int = 32,
    kappa: int = 32,
    max_new_tokens: int = 128,
    reward_normalizer_list=None,  # optional list of P RewardNormalizer
    binary_reward_fn=None,
    compute_regularizers: bool = False,
) -> list:
    """Run AISP on P prompts in parallel by pooling generation across the wave.

    Per-prompt state (u_hat, history) is kept independent. Generation of P*n
    perturbed samples per iteration runs in one batched call. Reward scoring
    flattens across the wave for one batched RM call. Importance-weighted
    update of u_hat is computed per-prompt (cheap, vectorised).

    Returns a list of P result dicts in the same shape as run_aisp_with_stored_rewards.
    """
    if compute_regularizers:
        # NOT YET IMPLEMENTED for the wave path — this function currently returns
        # token_kl_per_iter / fisher_per_iter as [None]*kappa. Running with
        # compute_regularizers here would silently produce all-None KL, so we fail
        # loudly instead. For now, run AISP with --wave_size 1 to get the uniform
        # deviation KL (the non-wave run_aisp_with_stored_rewards path).
        #
        # TO FIX (mirror run_aisp_with_stored_rewards):
        #   1. Before the iter loop, generate ONE set of clean u=0 base rollouts
        #      per prompt (zero perturbation), capturing per-trajectory pre-logits
        #      and an EOS-based valid mask -> base_kl_prelogits_gpu_list[p],
        #      base_kl_valid_gpu_list[p]  (use generate_with_perturbation_batch_wave
        #      with a zeros perturbation + capture_prelogits=True, split by prompt).
        #   2. In the per-prompt update loop, after u_hat[p] is updated, append
        #        token_kl_per_iter[p].append(
        #            kl_deviation(base_kl_prelogits_gpu_list[p], u_hat[p],
        #                         base_kl_valid_gpu_list[p], lm_head, device, model_dtype))
        #      so AISP-wave reports the SAME quantity as MISVO and non-wave AISP.
        #   3. (Optional) compute the per-trajectory Fisher the same way.
        raise NotImplementedError(
            "run_aisp_with_stored_rewards_wave does not compute regularizers "
            "(token_kl/fisher) yet — it would return all-None KL. Use --wave_size 1 "
            "for AISP with --compute_regularizers, or implement the base-rollout + "
            "kl_deviation logic here (see the TO FIX comment in this function).")
    P = len(prompt_ids_list)
    hidden_dim = llm.hidden_dim
    sigma = sigma_sq ** 0.5

    # Pad prompts and pre-compute one wave KV cache (re-used across all kappa iters).
    padded_ids, padded_mask, _ = llm.pad_and_stack_prompts(
        prompt_ids_list, attention_mask_list)
    prompt_decodes = [llm.decode(ids[0]) for ids in prompt_ids_list]
    prompt_kv_cache = llm.compute_prompt_kv_cache(padded_ids, padded_mask)

    # Per-prompt state.
    u_hat = torch.zeros(P, max_new_tokens, hidden_dim)
    r_best = [float("-inf")] * P
    y_best = [""] * P
    r_best_binary = [float("-inf")] * P
    y_best_binary = [""] * P
    mean_per_iter = [[] for _ in range(P)]
    best_per_iter = [[] for _ in range(P)]
    binary_mean_per_iter = [[] for _ in range(P)] if binary_reward_fn is not None else None
    binary_best_per_iter = [[] for _ in range(P)] if binary_reward_fn is not None else None
    binary_at_float_best_per_iter = [[] for _ in range(P)] if binary_reward_fn is not None else None
    eff_n_per_iter = [[] for _ in range(P)]
    l2_per_iter = [[] for _ in range(P)]
    gaussian_kl_per_iter = [[] for _ in range(P)]
    generations_per_step = [[] for _ in range(P)]

    iter_bar = tqdm(range(kappa), desc="  AISP wave iters", leave=False)
    for k in iter_bar:
        # Sample perturbations: V[p, i, t, d] = u_hat[p, t, d] + sigma * N(0, 1)
        v_batch = u_hat.unsqueeze(1) + sigma * torch.randn(P, n, max_new_tokens, hidden_dim)
        # Flatten to (P*n, T, d) for the wave generate call (consecutive rows = same prompt).
        perturbations = v_batch.reshape(P * n, max_new_tokens, hidden_dim)

        outputs = llm.generate_with_perturbation_batch_wave(
            padded_ids, padded_mask, perturbations, n_per_prompt=n,
            max_new_tokens=max_new_tokens,
            prompt_kv_cache=prompt_kv_cache,
            capture_prelogits=False,
        )
        full_texts = llm.batch_decode(outputs)

        # Strip prompt prefix per row (using the originating prompt's decoded text).
        responses_per_prompt = [[] for _ in range(P)]
        for row, ft in enumerate(full_texts):
            p = row // n
            responses_per_prompt[p].append(ft[len(prompt_decodes[p]):])

        # Flatten for one batched RM call across the entire wave.
        flat_prompts, flat_responses = [], []
        for p in range(P):
            for resp in responses_per_prompt[p]:
                flat_prompts.append(prompt_text_list[p])
                flat_responses.append(resp)
        flat_rewards = reward_fn_batch_multi(flat_prompts, flat_responses)

        # Per-prompt: split rewards back, run AISP update.
        for p in range(P):
            rewards_p = flat_rewards[p * n:(p + 1) * n]
            responses_p = responses_per_prompt[p]
            rewards_t = torch.tensor(rewards_p, dtype=torch.float32)

            # Generations log
            generations_per_step[p].append(
                [{"response": responses_p[i], "reward": rewards_p[i]} for i in range(n)])
            mean_per_iter[p].append(rewards_t.mean().item())
            best_at_k = rewards_t.max().item()
            best_per_iter[p].append(best_at_k)
            best_idx = int(rewards_t.argmax().item())
            if rewards_p[best_idx] > r_best[p]:
                r_best[p] = rewards_p[best_idx]
                y_best[p] = responses_p[best_idx]
            if binary_reward_fn is not None:
                bin_scores = [binary_reward_fn(prompt_text_list[p], r) for r in responses_p]
                binary_mean_per_iter[p].append(float(np.mean(bin_scores)))
                binary_best_per_iter[p].append(max(bin_scores))
                binary_at_float_best_per_iter[p].append(bin_scores[best_idx])
                bb_idx = int(np.argmax(bin_scores))
                if bin_scores[bb_idx] > r_best_binary[p]:
                    r_best_binary[p] = bin_scores[bb_idx]
                    y_best_binary[p] = responses_p[bb_idx]

            # Eq.18 importance update for prompt p.
            v_p = v_batch[p]  # (n, T, d)
            u_p = u_hat[p]    # (T, d)
            if reward_normalizer_list is not None and reward_normalizer_list[p] is not None:
                rewards_for_w = reward_normalizer_list[p].normalize(rewards_t)
            else:
                rewards_for_w = rewards_t
            dot_products = (u_p.unsqueeze(0) * v_p).sum(dim=(1, 2))  # (n,)
            scores = ((1.0 / lam) * rewards_for_w
                      - ((1.0 - alpha) / sigma_sq) * dot_products)
            weights = F.softmax(scores, dim=0)
            u_hat[p] = (weights.unsqueeze(1).unsqueeze(2) * v_p).sum(dim=0)
            eff_n_p = 1.0 / (weights ** 2).sum().item()
            eff_n_per_iter[p].append(eff_n_p)
            l2_val = (u_hat[p] ** 2).sum().item()
            l2_per_iter[p].append(l2_val)
            gaussian_kl_per_iter[p].append(l2_val / sigma_sq)

    # --- Final pass: one perturbation per prompt, generate, score ---
    v_final = u_hat + sigma * torch.randn(P, max_new_tokens, hidden_dim)  # (P, T, d)
    outputs_final = llm.generate_with_perturbation_batch_wave(
        padded_ids, padded_mask, v_final, n_per_prompt=1,
        max_new_tokens=max_new_tokens,
        prompt_kv_cache=prompt_kv_cache,
    )
    full_texts_final = llm.batch_decode(outputs_final)
    responses_final = [ft[len(prompt_decodes[p]):]
                       for p, ft in enumerate(full_texts_final)]
    rewards_final = reward_fn_batch_multi(prompt_text_list, responses_final)
    for p in range(P):
        if rewards_final[p] > r_best[p]:
            r_best[p] = rewards_final[p]
            y_best[p] = responses_final[p]

    results = []
    for p in range(P):
        res = {
            "best_response": y_best[p],
            "best_reward": r_best[p],
            "mean_per_iter": mean_per_iter[p],
            "best_per_iter": best_per_iter[p],
            "final_reward": rewards_final[p],
            "final_response": responses_final[p],
            "steering_vector": u_hat[p].detach().cpu(),
            "fisher_per_iter": [None] * kappa,
            "token_kl_per_iter": [None] * kappa,
            "gaussian_kl_per_iter": gaussian_kl_per_iter[p],
            "l2_per_iter": l2_per_iter[p],
            "eff_n_per_iter": eff_n_per_iter[p],
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
