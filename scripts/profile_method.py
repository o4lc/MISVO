"""
Detailed profiling for one (model, method) pair.

Captures, for a given config + method:
  - Wall-clock timing of model load, scorer load, prompt tokenization, and the
    method's per-prompt cost (mean / std / min / max over N prompts).
  - CUDA event timing (separate from wall-clock; matches GPU-side latency).
  - Peak GPU memory after each phase, and after each prompt.
  - PyTorch profiler trace on one prompt:
      * Chrome trace JSON (open with chrome://tracing or Perfetto).
      * Top-N op summary table (sorted by CUDA time).
      * `torch.cuda.memory_summary()` snapshot at peak.
  - Tokens/sec throughput estimate (generated tokens per prompt / per-prompt
    wall time).
  - Optional warmup pass to amortize HF first-call costs.

Usage (paper Table 4: 10 timed prompts after 5 warmup, single-prompt batches):
    python scripts/profile_method.py --config configs/lfm25_1b_shp.yaml --method misvo --num_prompts 10 --warmup 5
    python scripts/profile_method.py --config configs/lfm25_1b_shp.yaml --method aisp --num_prompts 10 --warmup 5
    python scripts/profile_method.py --config configs/lfm25_1b_shp.yaml --method bon --num_prompts 10 --warmup 5
    # Wave-batched profiling (pool 4 prompts per generate() call):
    python scripts/profile_method.py --config configs/lfm25_1b_shp.yaml --method misvo --num_prompts 3 --wave_size 4

`--num_prompts` is the number of timed *waves* (not raw prompts). With
`--wave_size W`, the script consumes `(warmup + num_prompts) * W` prompts.
"""

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.base_llm import BaseLLM
from src.models.reward_model import create_reward_model
from src.data.datasets import load_dataset_prompts, format_prompt, get_prompt_text
from src.utils import setup_logging, set_seed, logger


SUPPORTED_METHODS = ("aisp", "misvo", "bon")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def cuda_timer(label):
    """Context manager that times a block on CPU + GPU and tracks peak memory.

    Yields a dict that gets populated on exit with:
        wall_ms       — wall-clock ms (perf_counter)
        cuda_ms       — GPU-side ms (CUDA events; equals wall_ms when no CUDA)
        peak_mem_gb   — peak memory allocated during the block (GB, CUDA only)
        peak_mem_reserved_gb — peak reserved (cached) memory (GB)
    """
    info = {"label": label}
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)
        evt_start.record()
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if has_cuda:
            evt_end.record()
            torch.cuda.synchronize()
            cuda_ms = evt_start.elapsed_time(evt_end)
            peak = torch.cuda.max_memory_allocated()
            peak_res = torch.cuda.max_memory_reserved()
        else:
            cuda_ms = wall_ms
            peak = 0
            peak_res = 0
        info.update({
            "wall_ms": round(wall_ms, 2),
            "cuda_ms": round(cuda_ms, 2),
            "peak_mem_gb": round(peak / 1024**3, 3),
            "peak_mem_reserved_gb": round(peak_res / 1024**3, 3),
        })


def aggregate(samples, key):
    """Return mean / std / min / max of values[key] across `samples`."""
    vals = [s[key] for s in samples]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(vals)
    return {
        "n": n,
        "mean": round(mean, 3),
        "std": round(statistics.pstdev(vals) if n > 1 else 0.0, 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }


# ---------------------------------------------------------------------------
# Method dispatch — runs ONE prompt and returns the result dict
# ---------------------------------------------------------------------------

def _make_reward_fns(scorer):
    def reward_fn(p, r):
        return scorer.score(p, r)
    def reward_fn_batch(p, rs):
        return scorer.score_responses([p] * len(rs), rs)
    def reward_fn_batch_multi(prompts_flat, responses_flat):
        return scorer.score_responses(prompts_flat, responses_flat)
    return reward_fn, reward_fn_batch, reward_fn_batch_multi


def run_method_one_prompt(method, prompt, prompt_ids, attn_mask, prompt_text,
                          llm, scorer, cfg, max_new_tokens,
                          misvo_n_chunks=4, misvo_gen_chunks=1):
    """Run the method on a single prompt. Returns dict with at least
    'best_reward' and a 'gen_tokens' estimate (generated-token count for
    throughput calculations)."""
    reward_fn, reward_fn_batch, _ = _make_reward_fns(scorer)

    if method == "aisp":
        from src.methods.aisp import run_aisp_with_stored_rewards, RewardNormalizer
        c = cfg["aisp"]
        rn_cfg = cfg.get("reward_norm", {})
        normalizer = RewardNormalizer(mode=rn_cfg.get("mode", "none"),
                                      momentum=rn_cfg.get("momentum", 0.1))
        result = run_aisp_with_stored_rewards(
            llm=llm, prompt_ids=prompt_ids, attention_mask=attn_mask,
            reward_fn=reward_fn, prompt_text=prompt_text,
            n=c["n"], kappa=c["kappa"], sigma_sq=c["sigma_sq"],
            lam=c["lambda"], alpha=c["alpha"], max_new_tokens=max_new_tokens,
            reward_fn_batch=reward_fn_batch,
            reward_normalizer=normalizer,
        )
        # AISP samples n*kappa generations
        result["gen_tokens"] = c["n"] * c["kappa"] * max_new_tokens
        return result

    if method == "misvo":
        from src.methods.misvo import run_misvo
        c = cfg["misvo"]
        rn_cfg = cfg.get("reward_norm", {})
        from src.methods.aisp import RewardNormalizer
        normalizer = RewardNormalizer(mode=rn_cfg.get("mode", "none"),
                                      momentum=rn_cfg.get("momentum", 0.1))
        result = run_misvo(
            llm=llm, prompt_ids=prompt_ids, attention_mask=attn_mask,
            reward_fn=reward_fn, prompt_text=prompt_text,
            n_samples=c["n_samples"], n_steps=c["n_steps"],
            lr=c["lr"], lam=c["lambda"],
            temperature=c["temperature"], top_p=c["top_p"],
            max_new_tokens=max_new_tokens,
            reward_fn_batch=reward_fn_batch,
            reward_normalizer=normalizer,
            regularizer=c.get("regularizer", "frozen_fisher"),
            n_chunks=misvo_n_chunks,
            gen_chunks=misvo_gen_chunks,
            track_diagnostics=False,  # disable Fisher/KL tracking for fair perf measurement
        )
        result["gen_tokens"] = c["n_samples"] * c["n_steps"] * max_new_tokens
        return result

    if method == "bon":
        from src.methods.best_of_n import generate_bon_topp_responses
        c = cfg["bon"]
        responses = generate_bon_topp_responses(
            llm, prompt_ids, attn_mask, prompt_text,
            N=c["N"], max_new_tokens=max_new_tokens,
            temperature=c["temperature"], top_p=c["top_p"],
            batch_size=32,
        )
        scores = scorer.score_responses([prompt_text] * len(responses), responses)
        best = max(range(len(responses)), key=lambda i: scores[i])
        return {
            "best_response": responses[best],
            "best_reward": scores[best],
            "gen_tokens": c["N"] * max_new_tokens,
        }

    raise ValueError(f"Unsupported method: {method!r}")


# ---------------------------------------------------------------------------
# Wave dispatcher — runs the method on a batch of prompts in one wave call
# ---------------------------------------------------------------------------

WAVE_SUPPORTED_METHODS = ("aisp", "misvo", "bon")


def run_method_wave(method, prompt_packs, llm, scorer, cfg, max_new_tokens,
                    misvo_n_chunks=4):
    """Run the method on `len(prompt_packs)` prompts pooled in one wave call.

    `prompt_packs` is a list of (prompt, prompt_ids, attn_mask, prompt_text).
    Returns a list of per-prompt result dicts (same shape as the single-prompt
    function), plus an aggregate `gen_tokens` field on each.
    """
    _, _, reward_fn_batch_multi = _make_reward_fns(scorer)
    P = len(prompt_packs)
    ids_list = [pp[1] for pp in prompt_packs]
    mask_list = [pp[2] for pp in prompt_packs]
    text_list = [pp[3] for pp in prompt_packs]

    if method == "aisp":
        from src.methods.aisp import (
            run_aisp_with_stored_rewards_wave, RewardNormalizer)
        c = cfg["aisp"]
        rn_cfg = cfg.get("reward_norm", {})
        norm_list = [RewardNormalizer(mode=rn_cfg.get("mode", "none"),
                                      momentum=rn_cfg.get("momentum", 0.1))
                     for _ in range(P)]
        results = run_aisp_with_stored_rewards_wave(
            llm=llm,
            prompt_ids_list=ids_list,
            attention_mask_list=mask_list,
            prompt_text_list=text_list,
            reward_fn_batch_multi=reward_fn_batch_multi,
            n=c["n"], kappa=c["kappa"], sigma_sq=c["sigma_sq"],
            lam=c["lambda"], alpha=c["alpha"], max_new_tokens=max_new_tokens,
            reward_normalizer_list=norm_list,
        )
        gen_tokens = c["n"] * c["kappa"] * max_new_tokens
        for r in results:
            r["gen_tokens"] = gen_tokens
        return results

    if method == "misvo":
        from src.methods.misvo import run_misvo_wave
        from src.methods.aisp import RewardNormalizer
        c = cfg["misvo"]
        rn_cfg = cfg.get("reward_norm", {})
        norm_list = [RewardNormalizer(mode=rn_cfg.get("mode", "none"),
                                      momentum=rn_cfg.get("momentum", 0.1))
                     for _ in range(P)]
        results = run_misvo_wave(
            llm=llm,
            prompt_ids_list=ids_list,
            attention_mask_list=mask_list,
            prompt_text_list=text_list,
            reward_fn_batch_multi=reward_fn_batch_multi,
            n_samples=c["n_samples"], n_steps=c["n_steps"],
            lr=c["lr"], lam=c["lambda"],
            temperature=c["temperature"], top_p=c["top_p"],
            max_new_tokens=max_new_tokens,
            reward_normalizer_list=norm_list,
            regularizer=c.get("regularizer", "frozen_fisher"),
            n_chunks=misvo_n_chunks,
            track_diagnostics=False,
        )
        gen_tokens = c["n_samples"] * c["n_steps"] * max_new_tokens
        for r in results:
            r["gen_tokens"] = gen_tokens
        return results

    if method == "bon":
        from src.methods.best_of_n import generate_bon_topp_responses_wave
        c = cfg["bon"]
        responses_per_prompt = generate_bon_topp_responses_wave(
            llm, ids_list, mask_list,
            N=c["N"], max_new_tokens=max_new_tokens,
            temperature=c["temperature"], top_p=c["top_p"],
            batch_size=32 * P,
        )
        results = []
        # Score across the wave in one batched RM call.
        flat_prompts, flat_responses, owner_idx = [], [], []
        for p, rs in enumerate(responses_per_prompt):
            for r in rs:
                flat_prompts.append(text_list[p])
                flat_responses.append(r)
                owner_idx.append(p)
        flat_scores = scorer.score_responses(flat_prompts, flat_responses)
        for p, rs in enumerate(responses_per_prompt):
            scores_p = [flat_scores[i] for i in range(len(flat_scores))
                        if owner_idx[i] == p]
            best = max(range(len(rs)), key=lambda i: scores_p[i])
            results.append({
                "best_response": rs[best],
                "best_reward": scores_p[best],
                "gen_tokens": c["N"] * max_new_tokens,
            })
        return results

    raise ValueError(f"Wave path not supported for method: {method!r} "
                     f"(supported: {WAVE_SUPPORTED_METHODS})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detailed profiling of a single (model, method) pair")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (defines model, dataset, hyperparams)")
    parser.add_argument("--method", type=str, required=True, choices=SUPPORTED_METHODS,
                        help="Method to profile")
    parser.add_argument("--num_prompts", type=int, default=3,
                        help="How many prompts to time (default: 3)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup prompts to run before timing starts "
                             "(amortizes JIT/HF first-call cost; default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rm_device", type=str, default="cuda",
                        help="Reward model device (default: cuda)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: profiles/<model>/<method>/)")
    parser.add_argument("--no_torch_profiler", action="store_true", default=False,
                        help="Skip the chrome-trace + op-summary pass (only end-to-end timing)")
    parser.add_argument("--profile_prompt_idx", type=int, default=0,
                        help="Which prompt index to use for the torch.profiler trace (default: 0)")
    parser.add_argument("--misvo_n_chunks", type=int, default=4,
                        help="MISVO Phase 1 position chunking (default: 4; matches run_misvo.py)")
    parser.add_argument("--misvo_gen_chunks", type=int, default=1,
                        help="MISVO Phase 1 sampling rollout chunking (default: 1; matches run_misvo.py)")
    parser.add_argument("--wave_size", type=int, default=1,
                        help="Cross-prompt wave size. >1 pools that many prompts "
                             "into one generation call. Supported for: "
                             f"{WAVE_SUPPORTED_METHODS}. "
                             "Default 1 (no wave).")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    # --- Load config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    attn_implementation = cfg.get("generation", {}).get("attn_implementation")

    output_dir = args.output_dir or os.path.join(
        "profiles", f"{model_name}__{args.method}")
    os.makedirs(output_dir, exist_ok=True)

    wave_size = max(1, args.wave_size)
    if wave_size > 1 and args.method not in WAVE_SUPPORTED_METHODS:
        logger.warning(f"  --wave_size={wave_size} requested but method "
                       f"{args.method!r} has no wave entry point; "
                       f"falling back to per-prompt (wave_size=1).")
        wave_size = 1

    summary = {
        "config": args.config,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "method": args.method,
        "num_prompts": args.num_prompts,
        "warmup": args.warmup,
        "wave_size": wave_size,
        "max_new_tokens": max_new_tokens,
        "attn_implementation": attn_implementation,
        "misvo_n_chunks": args.misvo_n_chunks,
        "misvo_gen_chunks": args.misvo_gen_chunks,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    logger.info("=" * 70)
    logger.info(f"Profiling: model={model_name}, method={args.method}, "
                f"prompts={args.num_prompts} (warmup={args.warmup}, "
                f"wave_size={wave_size})")
    logger.info(f"Device: {summary['device_name']}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # --- Phase: Model load ---
    with cuda_timer("model_load") as t_load:
        llm = BaseLLM(model_name, attn_implementation=attn_implementation)
        llm.enable_thinking = cfg.get("generation", {}).get("enable_thinking")
        llm.load()
    summary["model_load"] = t_load
    logger.info(f"  Model load: wall={t_load['wall_ms']:.0f}ms, "
                f"peak={t_load['peak_mem_gb']:.2f}GB")

    # --- Phase: Scorer load (RM or rule-based) ---
    scoring_cfg = cfg.get("scoring", {})
    score_type = scoring_cfg.get("type")
    with cuda_timer("scorer_load") as t_sc:
        if score_type == "mbpp_plus":
            from src.evaluation.mbppplus_scorer import MBPPPlusScorer
            scorer = MBPPPlusScorer(timeout=scoring_cfg.get("timeout", 5))
        else:
            scorer = create_reward_model(cfg["reward_model"]["id"],
                                         devices=args.rm_device,
                                         dtype=torch.bfloat16)
        scorer.load()
    summary["scorer_load"] = t_sc
    logger.info(f"  Scorer load: wall={t_sc['wall_ms']:.0f}ms, "
                f"peak={t_sc['peak_mem_gb']:.2f}GB")

    # --- Load prompts ---
    # With wave_size > 1, each timed/warmup unit consumes `wave_size` prompts.
    needed = (args.num_prompts + args.warmup) * wave_size
    raw = load_dataset_prompts(dataset_name, max_samples=needed, seed=args.seed)
    raw = raw[:needed]
    prompts = [format_prompt(p, model_name, dataset_name) for p in raw]

    # Pre-encode all prompts (and time tokenization separately)
    encodings = []
    with cuda_timer("encode_all") as t_enc:
        for prompt in prompts:
            ids, mask = llm.encode_prompt(prompt)
            encodings.append((ids, mask, get_prompt_text(prompt)))
    summary["encode_all"] = t_enc
    summary["prompt_token_lengths"] = [int(ids.shape[1]) for ids, _, _ in encodings]
    logger.info(f"  Encoded {len(encodings)} prompts in "
                f"{t_enc['wall_ms']:.0f}ms (lengths: "
                f"min={min(summary['prompt_token_lengths'])}, "
                f"max={max(summary['prompt_token_lengths'])})")

    # ---- Helpers to invoke the method on one wave (or one prompt) ----
    def _invoke_on(start, end):
        """Run the method on encodings[start:end] (one wave). Returns a list of
        per-prompt result dicts."""
        if wave_size > 1:
            packs = [(prompts[j], encodings[j][0], encodings[j][1], encodings[j][2])
                     for j in range(start, end)]
            return run_method_wave(args.method, packs, llm, scorer, cfg,
                                   max_new_tokens,
                                   misvo_n_chunks=args.misvo_n_chunks)
        ids, mask, text = encodings[start]
        result = run_method_one_prompt(
            args.method, prompts[start], ids, mask, text,
            llm, scorer, cfg, max_new_tokens,
            misvo_n_chunks=args.misvo_n_chunks,
            misvo_gen_chunks=args.misvo_gen_chunks)
        return [result]

    # --- Warmup (one wave per warmup iteration) ---
    for w in range(args.warmup):
        s, e = w * wave_size, (w + 1) * wave_size
        if e > len(encodings):
            break
        plens = [int(encodings[j][0].shape[1]) for j in range(s, e)]
        logger.info(f"  Warmup wave {w+1}/{args.warmup} "
                    f"(prompts={list(range(s, e))}, plens={plens})...")
        with cuda_timer("warmup") as _:
            _invoke_on(s, e)

    # --- Timed waves ---
    per_prompt = []   # one entry per prompt (wave_time/wave_size for amortized)
    per_wave = []     # one entry per wave (raw timings)
    base_idx = args.warmup * wave_size
    n_waves = args.num_prompts  # treat num_prompts as number of timed waves
    for w in range(n_waves):
        s = base_idx + w * wave_size
        e = s + wave_size
        if e > len(encodings):
            logger.warning(f"  Not enough prompts for wave {w+1}; stopping.")
            break
        plens = [int(encodings[j][0].shape[1]) for j in range(s, e)]
        logger.info(f"  Profiling wave {w+1}/{n_waves} "
                    f"(prompts={list(range(s, e))}, plens={plens})...")
        with cuda_timer("wave") as t_w:
            results = _invoke_on(s, e)
        wave_gen_tokens = sum(r.get("gen_tokens", 0) for r in results)
        wave_rec = {
            "wave_idx": w,
            "prompt_indices": list(range(s, e)),
            "prompt_lens": plens,
            "gen_tokens_total": wave_gen_tokens,
            "tokens_per_sec": (wave_gen_tokens / (t_w["wall_ms"] / 1000.0)
                               if wave_gen_tokens and t_w["wall_ms"] > 0 else None),
            "best_rewards": [r.get("best_reward") for r in results],
        }
        wave_rec.update(t_w)
        per_wave.append(wave_rec)
        # Amortized per-prompt records: each prompt sees wave_time / wave_size.
        for k, j in enumerate(range(s, e)):
            rec = {
                "idx": j,
                "wave_idx": w,
                "wave_size": wave_size,
                "prompt_len": plens[k],
                "gen_tokens_total": results[k].get("gen_tokens", 0),
                "wall_ms": round(t_w["wall_ms"] / wave_size, 2),
                "cuda_ms": round(t_w["cuda_ms"] / wave_size, 2),
                "peak_mem_gb": t_w["peak_mem_gb"],
                "peak_mem_reserved_gb": t_w["peak_mem_reserved_gb"],
                "tokens_per_sec": (results[k].get("gen_tokens", 0) /
                                   ((t_w["wall_ms"] / wave_size) / 1000.0)
                                   if results[k].get("gen_tokens", 0) and t_w["wall_ms"] > 0
                                   else None),
                "best_reward": results[k].get("best_reward"),
            }
            per_prompt.append(rec)
        msg = (f"    wave wall={t_w['wall_ms']:.0f}ms, "
               f"cuda={t_w['cuda_ms']:.0f}ms, peak={t_w['peak_mem_gb']:.2f}GB"
               + (f", wave tok/s={wave_rec['tokens_per_sec']:.1f}"
                  if wave_rec['tokens_per_sec'] else "")
               + (f"  →  per-prompt amortized={t_w['wall_ms']/wave_size:.0f}ms"
                  if wave_size > 1 else ""))
        logger.info(msg)

    summary["per_prompt"] = per_prompt
    summary["per_wave"] = per_wave
    summary["aggregate"] = {
        "wall_ms": aggregate(per_prompt, "wall_ms"),
        "cuda_ms": aggregate(per_prompt, "cuda_ms"),
        "peak_mem_gb": aggregate(per_prompt, "peak_mem_gb"),
        "tokens_per_sec": aggregate(
            [{"v": r["tokens_per_sec"]} for r in per_prompt
             if r.get("tokens_per_sec") is not None],
            "v"),
    }
    if wave_size > 1:
        summary["aggregate_per_wave"] = {
            "wall_ms": aggregate(per_wave, "wall_ms"),
            "cuda_ms": aggregate(per_wave, "cuda_ms"),
            "peak_mem_gb": aggregate(per_wave, "peak_mem_gb"),
            "tokens_per_sec": aggregate(
                [{"v": r["tokens_per_sec"]} for r in per_wave
                 if r.get("tokens_per_sec") is not None],
                "v"),
        }

    # --- torch.profiler trace on one wave (or one prompt) ---
    if not args.no_torch_profiler and per_prompt:
        try:
            from torch.profiler import profile, ProfilerActivity, schedule
            # Profile the wave starting at args.profile_prompt_idx (clipped
            # to a valid wave start). args.profile_prompt_idx is interpreted
            # as a prompt index relative to the full encoded list.
            warmup_offset = args.warmup * wave_size
            max_start = len(encodings) - wave_size
            pidx = warmup_offset + args.profile_prompt_idx
            pidx = max(warmup_offset, min(pidx, max_start))
            wave_end = pidx + wave_size
            wave_indices = list(range(pidx, wave_end))
            wave_plens = [int(encodings[j][0].shape[1]) for j in wave_indices]
            logger.info(f"  torch.profiler trace on wave {wave_indices} "
                        f"(plens={wave_plens})...")
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with profile(activities=activities,
                         record_shapes=True,
                         profile_memory=True,
                         with_stack=False) as prof:
                _invoke_on(pidx, wave_end)
            chrome_path = os.path.join(output_dir, "chrome_trace.json")
            ops_path = os.path.join(output_dir, "op_summary.txt")
            mem_path = os.path.join(output_dir, "memory.txt")
            prof.export_chrome_trace(chrome_path)
            logger.info(f"  Wrote chrome trace → {chrome_path}")

            with open(ops_path, "w") as f:
                sort = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
                f.write("# Sorted by " + sort + "\n\n")
                f.write(prof.key_averages().table(sort_by=sort, row_limit=40))
                f.write("\n\n# Sorted by self_cpu_time_total\n\n")
                f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
            logger.info(f"  Wrote op summary  → {ops_path}")

            if torch.cuda.is_available():
                with open(mem_path, "w") as f:
                    f.write(torch.cuda.memory_summary(abbreviated=False))
                logger.info(f"  Wrote memory summary → {mem_path}")

            summary["torch_profiler"] = {
                "chrome_trace": chrome_path,
                "op_summary": ops_path,
                "memory_summary": mem_path if torch.cuda.is_available() else None,
                "profiled_prompt_idx": pidx,
                "profiled_wave_indices": wave_indices,
            }
        except Exception as e:
            logger.error(f"torch.profiler pass failed: {type(e).__name__}: {e}")
            summary["torch_profiler_error"] = f"{type(e).__name__}: {e}"

    # --- Write outputs ---
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(_render_report(summary))

    logger.info("=" * 70)
    logger.info(f"Wrote summary  → {summary_path}")
    logger.info(f"Wrote report   → {report_path}")
    logger.info("=" * 70)
    print()
    print(_render_report(summary))


def _render_report(s):
    """Human-readable report."""
    lines = []
    sep = "=" * 72
    wave_size = s.get("wave_size", 1)
    lines += [sep,
              f"Profile: model={s['model_name']}  method={s['method']}"
              + (f"  wave_size={wave_size}" if wave_size > 1 else ""),
              f"Config:  {s['config']}",
              f"Device:  {s['device_name']}  (torch {s['torch_version']})",
              f"Attn impl: {s.get('attn_implementation') or 'default'}",
              sep,
              ""]

    def _fmt(t, label):
        return (f"  {label:<22} wall={t['wall_ms']:>9.1f} ms"
                f"  cuda={t['cuda_ms']:>9.1f} ms"
                f"  peak={t['peak_mem_gb']:>5.2f} GB")

    lines += ["LOAD PHASES",
              "-" * 72,
              _fmt(s["model_load"], "model load"),
              _fmt(s["scorer_load"], "scorer load"),
              _fmt(s["encode_all"], f"encode {len(s['prompt_token_lengths'])} prompts"),
              ""]
    lengths = s["prompt_token_lengths"]
    lines += [f"  prompt token lengths: min={min(lengths)} max={max(lengths)} "
              f"mean={sum(lengths)/len(lengths):.0f}",
              ""]

    agg = s["aggregate"]
    lines += [f"PER-PROMPT TIMING ({len(s['per_prompt'])} prompts, "
              f"warmup={s['warmup']})",
              "-" * 72,
              f"  wall_ms      mean={agg['wall_ms']['mean']:>9.1f}   "
              f"std={agg['wall_ms']['std']:>9.1f}   "
              f"min={agg['wall_ms']['min']:>9.1f}   "
              f"max={agg['wall_ms']['max']:>9.1f}",
              f"  cuda_ms      mean={agg['cuda_ms']['mean']:>9.1f}   "
              f"std={agg['cuda_ms']['std']:>9.1f}   "
              f"min={agg['cuda_ms']['min']:>9.1f}   "
              f"max={agg['cuda_ms']['max']:>9.1f}",
              f"  peak_mem_gb  mean={agg['peak_mem_gb']['mean']:>9.2f}   "
              f"std={agg['peak_mem_gb']['std']:>9.2f}   "
              f"min={agg['peak_mem_gb']['min']:>9.2f}   "
              f"max={agg['peak_mem_gb']['max']:>9.2f}"]
    if agg["tokens_per_sec"]["n"] > 0:
        lines.append(
            f"  tokens/sec   mean={agg['tokens_per_sec']['mean']:>9.1f}   "
            f"std={agg['tokens_per_sec']['std']:>9.1f}   "
            f"min={agg['tokens_per_sec']['min']:>9.1f}   "
            f"max={agg['tokens_per_sec']['max']:>9.1f}")
    if wave_size > 1:
        lines.append(f"  (wall_ms/cuda_ms = wave_time / wave_size; "
                     f"see PER-WAVE TIMING below for raw wave times)")
    lines.append("")

    if wave_size > 1 and "aggregate_per_wave" in s:
        wagg = s["aggregate_per_wave"]
        lines += [f"PER-WAVE TIMING ({len(s['per_wave'])} waves of {wave_size})",
                  "-" * 72,
                  f"  wall_ms      mean={wagg['wall_ms']['mean']:>9.1f}   "
                  f"std={wagg['wall_ms']['std']:>9.1f}   "
                  f"min={wagg['wall_ms']['min']:>9.1f}   "
                  f"max={wagg['wall_ms']['max']:>9.1f}",
                  f"  cuda_ms      mean={wagg['cuda_ms']['mean']:>9.1f}   "
                  f"std={wagg['cuda_ms']['std']:>9.1f}   "
                  f"min={wagg['cuda_ms']['min']:>9.1f}   "
                  f"max={wagg['cuda_ms']['max']:>9.1f}",
                  f"  peak_mem_gb  mean={wagg['peak_mem_gb']['mean']:>9.2f}   "
                  f"std={wagg['peak_mem_gb']['std']:>9.2f}   "
                  f"min={wagg['peak_mem_gb']['min']:>9.2f}   "
                  f"max={wagg['peak_mem_gb']['max']:>9.2f}"]
        if wagg["tokens_per_sec"]["n"] > 0:
            lines.append(
                f"  wave_tok/s   mean={wagg['tokens_per_sec']['mean']:>9.1f}   "
                f"std={wagg['tokens_per_sec']['std']:>9.1f}   "
                f"min={wagg['tokens_per_sec']['min']:>9.1f}   "
                f"max={wagg['tokens_per_sec']['max']:>9.1f}")
        lines.append("")

    lines += ["PER-PROMPT DETAIL", "-" * 72,
              f"  {'idx':>4}  {'plen':>5}  {'wall_ms':>9}  {'cuda_ms':>9}  "
              f"{'peak_GB':>7}  {'tok/s':>7}  {'best_R':>8}"]
    for r in s["per_prompt"]:
        toks = f"{r['tokens_per_sec']:.1f}" if r.get("tokens_per_sec") else "    -"
        rew = f"{r['best_reward']:.3f}" if r.get("best_reward") is not None else "    -"
        lines.append(f"  {r['idx']:>4}  {r['prompt_len']:>5}  "
                     f"{r['wall_ms']:>9.1f}  {r['cuda_ms']:>9.1f}  "
                     f"{r['peak_mem_gb']:>7.2f}  {toks:>7}  {rew:>8}")
    lines.append("")

    if "torch_profiler" in s:
        lines += ["TORCH PROFILER ARTIFACTS",
                  "-" * 72,
                  f"  chrome trace : {s['torch_profiler']['chrome_trace']}",
                  f"  op summary   : {s['torch_profiler']['op_summary']}",
                  f"  memory dump  : {s['torch_profiler'].get('memory_summary') or 'n/a'}",
                  ""]
    lines.append(sep)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
