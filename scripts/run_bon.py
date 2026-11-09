"""
Best-of-N experiment runner.

Two-stage approach:
Stage 1: Load base LLM, generate N responses per prompt, save to disk
Stage 2: Load reward model, score all responses, select best

Usage:
    python scripts/run_bon.py --config configs/gemma3_4b_it_shp.yaml --variant topp --num_prompts 100
"""

import argparse
import json
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.base_llm import BaseLLM
from src.models.reward_model import create_reward_model
from src.data.datasets import load_dataset_prompts, format_prompt, get_prompt_text
from src.methods.best_of_n import (
    generate_bon_topp_responses,
    generate_bon_topp_responses_wave,
)
from src.utils import (setup_logging, set_seed, save_responses, save_generations, incremental_save, try_resume, logger, free_memory,
                       add_shard_args, resolve_shard_args, compute_shard_slice, shard_output_path,
                       resolve_range_args, range_output_path,
                       errors_path as _errors_path, append_worker_error as _append_worker_error,
                       merge_worker_errors as _merge_worker_errors, save_errors as _save_errors,
                       load_existing_errors as _load_existing_errors, make_error_entry as _make_error_entry)


def _bon_worker(gpu_id, work_queue, prompts, result_queue,
                cfg=None, model_name=None,
                max_new_tokens=128, N=64, batch_size=32,
                temperature=0.7, top_p=0.95, seed=42,
                attn_implementation=None,
                output_path=None, wave_size=1):
    """Multi-GPU worker for BoN generation stage."""
    setup_logging()
    set_seed(seed + gpu_id)
    device = f"cuda:{gpu_id}"
    logger.info(f"[GPU {gpu_id}] Loading LLM on {device}...")

    llm = BaseLLM(model_name, device=device,
                  attn_implementation=attn_implementation)
    llm.enable_thinking = cfg.get("generation", {}).get("enable_thinking")
    llm.load()

    import queue

    def _process_one(idx):
        prompt = prompts[idx]
        prompt_text = get_prompt_text(prompt)
        prompt_ids, attn_mask = llm.encode_prompt(prompt)
        try:
            responses = generate_bon_topp_responses(
                llm, prompt_ids, attn_mask, prompt_text,
                N=N, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Prompt {idx+1}/{len(prompts)} FAILED: "
                         f"{type(e).__name__}: {e}")
            _append_worker_error(output_path, gpu_id, idx, prompt_text, e)
            torch.cuda.empty_cache()
            free_memory()
            return None
        return {"prompt": prompt_text, "responses": responses}

    def _process_wave(indices):
        if len(indices) <= 1:
            return [(i, _process_one(i)) for i in indices]
        prompt_text_list, ids_list, mask_list = [], [], []
        for i in indices:
            pr = prompts[i]
            prompt_text_list.append(get_prompt_text(pr))
            ids, mask = llm.encode_prompt(pr)
            ids_list.append(ids)
            mask_list.append(mask)
        try:
            responses_per_prompt = generate_bon_topp_responses_wave(
                llm, ids_list, mask_list,
                N=N, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.warning(f"[GPU {gpu_id}] BoN wave of {len(indices)} prompts FAILED "
                           f"({type(e).__name__}: {e}); falling back to per-prompt")
            torch.cuda.empty_cache()
            free_memory()
            return [(i, _process_one(i)) for i in indices]
        return [(idx, {"prompt": prompt_text_list[k], "responses": responses_per_prompt[k]})
                for k, idx in enumerate(indices)]

    n_completed = 0
    while True:
        pending_idx = []
        for _ in range(max(1, wave_size)):
            try:
                pending_idx.append(work_queue.get(timeout=2))
            except queue.Empty:
                break
        if not pending_idx:
            break
        if len(pending_idx) > 1:
            logger.info(f"[GPU {gpu_id}] BoN wave of {len(pending_idx)} prompts × {N} candidates "
                        f"({pending_idx[0]+1}..{pending_idx[-1]+1}/{len(prompts)})")
        else:
            logger.info(f"[GPU {gpu_id}] Prompt {pending_idx[0]+1}/{len(prompts)}: "
                        f"generating {N} candidates")
        for idx, entry in _process_wave(pending_idx):
            result_queue.put((idx, entry))
            if entry is not None:
                n_completed += 1
        free_memory()

    llm.unload()
    logger.info(f"[GPU {gpu_id}] Done ({n_completed} prompts)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True, choices=["topp"])
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wave_size", type=int, default=None,
                        help="Pool this many prompts per generate() call. "
                             "Falls back to per-prompt on OOM. Default 1 (no wave).")
    parser.add_argument("--rm_device", type=str, default="cuda",
                        help="Device(s) for reward model: 'cuda', 'cpu', 'cuda:1', or 'cuda:1,cuda:2' for multi-GPU")
    parser.add_argument("--incremental_save", action="store_true", default=False,
                        help="Save results to disk after each prompt (atomic writes)")
    parser.add_argument("--float_rewards", action="store_true", default=None,
                        help="Use float rewards for selection (overrides config)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel prompt generation. "
                             "Stage 1 (generate) shards prompts across workers; "
                             "Stage 2 (score) always runs single-process.")
    add_shard_args(parser)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    attn_implementation = cfg.get("generation", {}).get("attn_implementation", None)

    variant_tag = "topp"

    scoring_cfg = cfg.get("scoring", {})
    use_float_rewards = (args.float_rewards if args.float_rewards is not None else scoring_cfg.get("float_rewards", False)) and scoring_cfg.get("type") == "mbpp_plus"

    # Read variant-specific config for tag and later use
    float_suffix = "_floatrw" if use_float_rewards else ""
    bon_cfg = cfg["bon"]
    N = args.N or bon_cfg["N"]
    temperature = bon_cfg["temperature"]
    top_p = bon_cfg["top_p"]
    wave_size = args.wave_size if args.wave_size is not None else bon_cfg.get("wave_size", 1)
    output_tag = (f"bon_{variant_tag}_{model_name}_{dataset_name}"
                  f"_N{N}_t{temperature}_p{top_p}"
                  f"{float_suffix}"
                  f"_seed{args.seed}")

    candidates_path = os.path.join(args.output_dir, f"{output_tag}_candidates.json")
    output_path = os.path.join(args.output_dir, f"{output_tag}.json")

    # --- Partition: explicit range mode OR equal-slice shard mode ---
    # Output paths are renamed early; prompt slicing happens later in the
    # generation stage, once prompts are loaded.
    range_start, range_end = resolve_range_args(args, args.num_prompts)
    shard_id = num_shards = None
    if range_start is not None:
        candidates_path = range_output_path(candidates_path, range_start, range_end)
        output_path = range_output_path(output_path, range_start, range_end)
        if not args.incremental_save:
            logger.info("Forcing --incremental_save for range execution")
            args.incremental_save = True
    else:
        shard_id, num_shards = resolve_shard_args(args)
        if shard_id is not None:
            candidates_path = shard_output_path(candidates_path, shard_id, num_shards)
            output_path = shard_output_path(output_path, shard_id, num_shards)
            if not args.incremental_save:
                logger.info("Forcing --incremental_save for sharded execution")
                args.incremental_save = True

    # ---------------- Stage 1: generate candidates ----------------
    raw_prompts = load_dataset_prompts(dataset_name, max_samples=args.num_prompts, seed=args.seed)
    prompts = [format_prompt(p, model_name, dataset_name) for p in raw_prompts[:args.num_prompts]]

    if range_start is not None:
        logger.info(f"Range mode [{range_start}, {range_end}): "
                    f"processing {range_end - range_start} of {len(prompts)} prompts")
        prompts = prompts[range_start:range_end]
        raw_prompts = raw_prompts[range_start:range_end]
        if not prompts:
            logger.info(f"Range [{range_start}, {range_end}): no prompts assigned, exiting")
            save_responses([], candidates_path)
            save_responses([], output_path)
            return
    elif shard_id is not None:
        shard_start, shard_end = compute_shard_slice(len(prompts), shard_id, num_shards)
        logger.info(f"Shard {shard_id}/{num_shards}: processing prompts [{shard_start}, {shard_end}) "
                    f"({shard_end - shard_start} of {len(prompts)} total)")
        prompts = prompts[shard_start:shard_end]
        raw_prompts = raw_prompts[shard_start:shard_end]
        if not prompts:
            logger.info(f"Shard {shard_id}: no prompts assigned, exiting")
            save_responses([], candidates_path)
            save_responses([], output_path)
            return

    logger.info(f"BoN(top-p): N={N}, temp={temperature}, top_p={top_p}")

    all_candidates = try_resume(candidates_path) if args.incremental_save else []
    start_idx = len(all_candidates)
    if start_idx > 0:
        logger.info(f"Skipping {start_idx} already-generated prompts")

    # --- Multi-GPU generation. The worker only generates candidates
    # (no scorer involved), so this is safe for all scoring types
    # including SHP/RM-based runs. Scoring (Stage 2) still runs
    # single-process on the merged candidates file below.
    if args.num_gpus > 1 or wave_size > 1:
        logger.info(f"Worker mode: {args.num_gpus} GPU(s), wave_size={wave_size} (generation only)")
        from src.multi_gpu import run_multi_gpu
        worker_kwargs = dict(
            cfg=cfg, model_name=model_name,
            max_new_tokens=max_new_tokens, N=N, batch_size=args.batch_size,
            seed=args.seed,
            attn_implementation=attn_implementation,
            progress_desc=f"BoN({variant_tag})",
        )
        worker_kwargs["temperature"] = temperature
        worker_kwargs["top_p"] = top_p
        worker_kwargs["output_path"] = candidates_path
        worker_kwargs["wave_size"] = wave_size
        all_candidates = run_multi_gpu(
            num_gpus=args.num_gpus,
            worker_fn=_bon_worker,
            prompts=prompts,
            start_idx=start_idx,
            existing_results=all_candidates,
            output_path=candidates_path,
            worker_kwargs=worker_kwargs,
        )
        _merge_worker_errors(candidates_path, args.num_gpus)
        logger.info(f"Saved candidates to {candidates_path}")
    else:
        # --- Single-GPU path ---
        llm = BaseLLM(model_name,
                      attn_implementation=attn_implementation)
        llm.enable_thinking = cfg.get("generation", {}).get("enable_thinking")
        llm.load()

        errors = _load_existing_errors(candidates_path)

        for idx, prompt in enumerate(tqdm(prompts[start_idx:], desc=f"BoN({variant_tag})", initial=start_idx, total=len(prompts)), start=start_idx):
            prompt_text = get_prompt_text(prompt)
            logger.info(f"Prompt {idx+1}/{len(prompts)}: generating {N} candidates")
            prompt_ids, attn_mask = llm.encode_prompt(prompt)

            try:
                responses = generate_bon_topp_responses(
                    llm, prompt_ids, attn_mask, prompt_text,
                    N=N, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                logger.error(f"Prompt {idx+1}/{len(prompts)} FAILED, skipping: "
                             f"{type(e).__name__}: {e}")
                errors.append(_make_error_entry(idx, prompt_text, e))
                _save_errors(candidates_path, errors)
                torch.cuda.empty_cache()
                free_memory()
                continue

            all_candidates.append({
                "prompt": prompt_text,
                "responses": responses,
            })
            if args.incremental_save:
                incremental_save(all_candidates, candidates_path)

        llm.unload()
        save_responses(all_candidates, candidates_path)
        logger.info(f"Saved candidates to {candidates_path}")
        _save_errors(candidates_path, errors)
        if errors:
            logger.warning(f"{len(errors)} prompts failed; see {_errors_path(candidates_path)}")

    # ---------------- Stage 2: score candidates ----------------
    # Load candidates
    with open(candidates_path, "r") as f:
        all_candidates = json.load(f)

    # Load scorer: MBPP+ or reward model
    if scoring_cfg.get("type") == "mbpp_plus":
        from src.evaluation.mbppplus_scorer import MBPPPlusScorer
        scorer = MBPPPlusScorer(
            timeout=scoring_cfg.get("timeout", 5),
        )
        scorer.load()
    else:
        scorer = create_reward_model(cfg["reward_model"]["id"], devices=args.rm_device)
        scorer.load()

    results = try_resume(output_path) if args.incremental_save else []
    score_start_idx = len(results)
    if score_start_idx > 0:
        logger.info(f"Skipping {score_start_idx} already-scored prompts")
    if use_float_rewards:
        logger.info("Using float rewards for selection (binary for ground truth)")
    for idx, cand_entry in enumerate(tqdm(all_candidates[score_start_idx:], desc="Scoring", initial=score_start_idx, total=len(all_candidates)), start=score_start_idx):
        prompt = cand_entry["prompt"]
        responses = cand_entry["responses"]

        if use_float_rewards:
            # Score both in a single parallel pass
            binary_scores, float_scores = scorer.score_responses_both(
                [prompt] * len(responses), responses)
            best_idx = int(np.argmax(float_scores))
            best_binary_idx = int(np.argmax(binary_scores))
            generations = [{"response": responses[i], "reward": float_scores[i]}
                           for i in range(len(responses))]
            entry = {
                "prompt": prompt,
                "response": responses[best_binary_idx],
                "reward": binary_scores[best_binary_idx],
                "float_reward": float_scores[best_idx],
                "all_rewards": binary_scores,
                "all_float_rewards": float_scores,
                "generations_per_step": [generations],
            }
        else:
            scores = scorer.score_responses(
                [prompt] * len(responses), responses, batch_size=args.batch_size
            )
            best_idx = int(np.argmax(scores))
            generations = [{"response": responses[i], "reward": scores[i]}
                           for i in range(len(responses))]
            entry = {
                "prompt": prompt,
                "response": responses[best_idx],
                "reward": scores[best_idx],
                "all_rewards": scores,
                "generations_per_step": [generations],
            }
        results.append(entry)
        if args.incremental_save:
            incremental_save(results, output_path)

    scorer.unload()
    save_generations(results, output_path)
    save_responses(results, output_path)

    rewards = [r["reward"] for r in results]
    logger.info(f"BoN({variant_tag}) mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
