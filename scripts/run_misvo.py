"""
Main MISVO experiment runner.

Default: LLM on GPU, RM on CPU. Use --rm_device cuda to put RM on GPU
(requires >=24GB VRAM for Llama-3.2-3B, >=40GB for larger models).

Usage:
    python scripts/run_misvo.py --config configs/lfm25_1b_shp.yaml --num_prompts 100 --seed 42
    python scripts/run_misvo.py --config configs/lfm25_1b_shp.yaml --num_prompts 100 --rm_device cuda
"""

import argparse
import json
import os
import sys
import yaml
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.base_llm import BaseLLM
from src.models.reward_model import create_reward_model
from src.data.datasets import load_dataset_prompts, format_prompt, get_prompt_text
from src.methods.misvo import run_misvo, run_misvo_wave
from src.methods.aisp import RewardNormalizer
from src.utils import (setup_logging, set_seed, save_responses, save_generations, incremental_save, try_resume, logger, free_memory,
                       add_shard_args, resolve_shard_args, compute_shard_slice, shard_output_path,
                       resolve_range_args, range_output_path,
                       errors_path as _errors_path, append_worker_error as _append_worker_error,
                       merge_worker_errors as _merge_worker_errors, save_errors as _save_errors,
                       load_existing_errors as _load_existing_errors, make_error_entry as _make_error_entry)


def _misvo_worker(gpu_id, work_queue, prompts, result_queue,
                  cfg=None, use_float_rewards=False, model_name=None,
                  max_new_tokens=128, n_samples=16, n_steps=20, lr=0.01,
                  lam=1.0, temperature=0.7, top_p=0.95,
                  regularizer="frozen_fisher",
                  reward_norm_mode="none", reward_norm_momentum=0.1, seed=42,
                  save_vectors=False, vectors_dir=None,
                  attn_implementation=None, n_chunks=4,
                  gen_chunks=1, output_path=None, wave_size=1):
    """Multi-GPU worker: loads LLM + scorer on assigned GPU, processes prompts."""
    import queue
    setup_logging()
    set_seed(seed + gpu_id)
    device = f"cuda:{gpu_id}"
    logger.info(f"[GPU {gpu_id}] Loading LLM on {device}...")

    llm = BaseLLM(model_name, device=device,
                  attn_implementation=attn_implementation)
    llm.enable_thinking = cfg.get("generation", {}).get("enable_thinking")
    llm.load()

    scoring_cfg = cfg.get("scoring", {})
    score_type = scoring_cfg.get("type")
    if score_type == "mbpp_plus":
        from src.evaluation.mbppplus_scorer import MBPPPlusScorer
        scorer = MBPPPlusScorer(timeout=scoring_cfg.get("timeout", 5))
        scorer.load()
    else:
        # RM-based scoring (e.g., SHP). Co-load the RM on this worker's GPU.
        # use_float_rewards has no meaning for RMs — force off so we use score().
        use_float_rewards = False
        scorer = create_reward_model(
            cfg["reward_model"]["id"], devices=device, dtype=torch.bfloat16)
        scorer.load()

    if use_float_rewards:
        def reward_fn(prompt_text, response_text):
            return scorer.score_float(prompt_text, response_text)
        def reward_fn_batch(prompt_text, responses):
            return scorer.score_responses_float(
                [prompt_text] * len(responses), responses)
    else:
        def reward_fn(prompt_text, response_text):
            return scorer.score(prompt_text, response_text)
        def reward_fn_batch(prompt_text, responses):
            return scorer.score_responses(
                [prompt_text] * len(responses), responses)

    reward_normalizer = RewardNormalizer(mode=reward_norm_mode, momentum=reward_norm_momentum)

    # ---- Helper: build the result entry for one (idx, result) pair ----
    def _build_entry(idx, prompt_text, result):
        entry = {
            "prompt": prompt_text,
            "response": result["best_response"],
            "mean_per_iter": result["mean_per_iter"],
            "best_per_iter": result["best_per_iter"],
            "final_reward": result["final_reward"],
            "fisher_per_iter": result["fisher_per_iter"],
            "fully_adaptive_fisher_per_iter": result["fully_adaptive_fisher_per_iter"],
            "kl_per_iter": result["kl_per_iter"],
            "l2_per_iter": result["l2_per_iter"],
            "u_norm_per_iter": result.get("u_norm_per_iter"),
        }
        if "binary_mean_per_iter" in result:
            entry["binary_mean_per_iter"] = result["binary_mean_per_iter"]
            entry["binary_best_per_iter"] = result["binary_best_per_iter"]
            entry["binary_at_float_best_per_iter"] = result.get("binary_at_float_best_per_iter")
        if use_float_rewards:
            entry["reward"] = result.get("best_binary_reward",
                                         scorer.score(prompt_text, result["best_response"]))
            entry["float_reward"] = result["best_reward"]
            entry["response"] = result.get("best_binary_response", result["best_response"])
        else:
            entry["reward"] = result["best_reward"]
        if save_vectors and vectors_dir is not None:
            vec_path = os.path.join(vectors_dir, f"v_{idx:05d}.pt")
            torch.save(result["steering_vector"], vec_path)
        entry["generations_per_step"] = result.get("generations_per_step", [])
        return entry

    # ---- Helper: process one prompt via the existing per-prompt path ----
    def _process_one(idx):
        prompt = prompts[idx]
        prompt_text = get_prompt_text(prompt)
        prompt_ids, attn_mask = llm.encode_prompt(prompt)
        binary_reward_fn_for_tracking = None
        if use_float_rewards:
            binary_reward_fn_for_tracking = lambda p, r: scorer.score(p, r)
        try:
            result = run_misvo(
                llm=llm, prompt_ids=prompt_ids, attention_mask=attn_mask,
                reward_fn=reward_fn, prompt_text=prompt_text,
                lam=lam, n_samples=n_samples, n_steps=n_steps, lr=lr,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, reward_fn_batch=reward_fn_batch,
                reward_normalizer=reward_normalizer,
                regularizer=regularizer,
                binary_reward_fn=binary_reward_fn_for_tracking,
                n_chunks=n_chunks,
                gen_chunks=gen_chunks,
            )
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Prompt {idx+1}/{len(prompts)} FAILED: "
                         f"{type(e).__name__}: {e}")
            _append_worker_error(output_path, gpu_id, idx, prompt_text, e)
            torch.cuda.empty_cache()
            free_memory()
            return None
        return _build_entry(idx, prompt_text, result)

    # ---- Helper: process a batch of prompts via wave path; on failure
    # falls back to per-prompt processing for the same indices ----
    def _process_wave(indices):
        if len(indices) <= 1:
            return [(i, _process_one(i)) for i in indices]
        # Encode all wave members.
        prompt_text_list, ids_list, mask_list = [], [], []
        for i in indices:
            pr = prompts[i]
            prompt_text_list.append(get_prompt_text(pr))
            ids, mask = llm.encode_prompt(pr)
            ids_list.append(ids)
            mask_list.append(mask)
        binary_reward_fn_for_tracking = None
        if use_float_rewards:
            binary_reward_fn_for_tracking = lambda p, r: scorer.score(p, r)
        # Wave reward function: scorer accepts heterogeneous prompt list directly.
        def reward_fn_batch_multi(prompts_flat, responses_flat):
            if use_float_rewards:
                return scorer.score_responses_float(prompts_flat, responses_flat)
            return scorer.score_responses(prompts_flat, responses_flat)
        # Independent normalizers per wave member (so EMA stats don't entangle).
        norm_list = [RewardNormalizer(mode=reward_norm_mode, momentum=reward_norm_momentum)
                     for _ in indices]
        try:
            results = run_misvo_wave(
                llm=llm,
                prompt_ids_list=ids_list,
                attention_mask_list=mask_list,
                prompt_text_list=prompt_text_list,
                reward_fn_batch_multi=reward_fn_batch_multi,
                lam=lam, n_samples=n_samples, n_steps=n_steps, lr=lr,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
                reward_normalizer_list=norm_list,
                regularizer=regularizer,
                binary_reward_fn=binary_reward_fn_for_tracking,
                n_chunks=n_chunks,
            )
        except Exception as e:
            logger.warning(f"[GPU {gpu_id}] Wave of {len(indices)} prompts FAILED "
                           f"({type(e).__name__}: {e}); falling back to per-prompt")
            torch.cuda.empty_cache()
            free_memory()
            return [(i, _process_one(i)) for i in indices]
        return [(idx, _build_entry(idx, prompt_text_list[k], results[k]))
                for k, idx in enumerate(indices)]

    n_completed = 0
    while True:
        # Pull up to wave_size indices, breaking out when the queue is empty.
        pending_idx = []
        for _ in range(max(1, wave_size)):
            try:
                pending_idx.append(work_queue.get(timeout=2))
            except queue.Empty:
                break
        if not pending_idx:
            break
        if len(pending_idx) > 1:
            logger.info(f"[GPU {gpu_id}] Wave of {len(pending_idx)} prompts "
                        f"({pending_idx[0]+1}..{pending_idx[-1]+1}/{len(prompts)})")
        else:
            logger.info(f"[GPU {gpu_id}] Prompt {pending_idx[0]+1}/{len(prompts)}")
        for idx, entry in _process_wave(pending_idx):
            result_queue.put((idx, entry))
            if entry is not None:
                n_completed += 1
        free_memory()

    llm.unload()
    scorer.unload()
    logger.info(f"[GPU {gpu_id}] Done ({n_completed} prompts)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    # Override hyperparameters
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--regularizer", type=str, default=None,
                        choices=["frozen_fisher", "steered_traj_fisher",
                                 "fully_adaptive_fisher", "l2", "kl"],
                        help="Regularizer: 'frozen_fisher' (default; MISVO, frozen-reference "
                             "Fisher), 'steered_traj_fisher' (on-policy rollouts, reference Fisher), "
                             "'fully_adaptive_fisher' (on-policy rollouts, on-policy Fisher), "
                             "'l2' (hidden-space L2), or 'kl' (direct KL). Use --lam 0 for "
                             "unregularized steering.")
    parser.add_argument("--rm_device", type=str, default="cpu",
                        help="Device(s) for reward model: 'cpu', 'cuda', 'cuda:1', or 'cuda:1,cuda:2' for multi-GPU")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for reward model scoring (reduce for large RMs)")
    parser.add_argument("--reward_norm", type=str, default=None,
                        choices=["none", "batch", "running"],
                        help="Reward normalization mode (overrides config)")
    parser.add_argument("--reward_norm_momentum", type=float, default=None,
                        help="Momentum for running reward normalization (overrides config)")
    parser.add_argument("--incremental_save", action="store_true", default=False,
                        help="Save results to disk after each prompt (atomic writes)")
    parser.add_argument("--save_vectors", action="store_true", default=False,
                        help="Save per-prompt steering vectors to a separate .pt file")
    parser.add_argument("--float_rewards", action="store_true", default=None,
                        help="Use float rewards for optimization (overrides config)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel prompt processing (one worker per GPU)")
    parser.add_argument("--n_chunks", type=int, default=4,
                        help="MISVO Phase 1 position chunking (1=no chunking; "
                             "raise if logits_u (n*T*V fp32) doesn't fit in VRAM)")
    parser.add_argument("--gen_chunks", type=int, default=1,
                        help="MISVO Phase 1 sampling rollout chunking (1=full batch; "
                             "raise if HF generate() with n_samples in one batch OOMs). "
                             "Auto-increased further on OOM during sampling.")
    parser.add_argument("--wave_size", type=int, default=None,
                        help="Process this many prompts in parallel via wave-batched "
                             "generation. Falls back to per-prompt on OOM. Default 1 (no wave).")
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

    # MISVO hyperparameters (config or override)
    misvo_cfg = cfg["misvo"]
    n_samples = args.n_samples or misvo_cfg["n_samples"]
    n_steps = args.n_steps or misvo_cfg["n_steps"]
    lr = args.lr if args.lr is not None else misvo_cfg["lr"]
    lam = args.lam if args.lam is not None else misvo_cfg["lambda"]
    temperature = args.temperature if args.temperature is not None else misvo_cfg["temperature"]
    top_p = args.top_p if args.top_p is not None else misvo_cfg["top_p"]
    regularizer = args.regularizer if args.regularizer is not None else misvo_cfg.get("regularizer", "frozen_fisher")
    wave_size = args.wave_size if args.wave_size is not None else misvo_cfg.get("wave_size", 1)
    # Reward normalization (CLI overrides config)
    reward_norm_cfg = cfg.get("reward_norm", {})
    reward_norm_mode = args.reward_norm if args.reward_norm is not None else reward_norm_cfg.get("mode", "none")
    reward_norm_momentum = args.reward_norm_momentum if args.reward_norm_momentum is not None else reward_norm_cfg.get("momentum", 0.1)
    reward_normalizer = RewardNormalizer(mode=reward_norm_mode, momentum=reward_norm_momentum)

    scoring_cfg = cfg.get("scoring", {})
    use_float_rewards = args.float_rewards if args.float_rewards is not None else scoring_cfg.get("float_rewards", False)

    output_tag = (f"misvo_{model_name}_{dataset_name}"
                  f"_ns{n_samples}_steps{n_steps}_lr{lr}_lam{lam}"
                  f"_reg{regularizer}"
                  f"_norm{reward_norm_mode}"
                  f"{'_floatrw' if use_float_rewards else ''}"
                  f"_seed{args.seed}")
    output_path = os.path.join(args.output_dir, f"{output_tag}.json")

    # Load prompts
    raw_prompts = load_dataset_prompts(dataset_name, max_samples=args.num_prompts, seed=args.seed)
    prompts = [format_prompt(p, model_name, dataset_name) for p in raw_prompts[:args.num_prompts]]

    # --- Partition: explicit range mode OR equal-slice shard mode ---
    range_start, range_end = resolve_range_args(args, len(prompts))
    if range_start is not None:
        logger.info(f"Range mode [{range_start}, {range_end}): "
                    f"processing {range_end - range_start} of {len(prompts)} prompts")
        prompts = prompts[range_start:range_end]
        raw_prompts = raw_prompts[range_start:range_end]
        output_path = range_output_path(output_path, range_start, range_end)
        if not args.incremental_save:
            logger.info("Forcing --incremental_save for range execution")
            args.incremental_save = True
        if not prompts:
            logger.info(f"Range [{range_start}, {range_end}): no prompts assigned, exiting")
            save_responses([], output_path)
            return
    else:
        shard_id, num_shards = resolve_shard_args(args)
        if shard_id is not None:
            shard_start, shard_end = compute_shard_slice(len(prompts), shard_id, num_shards)
            logger.info(f"Shard {shard_id}/{num_shards}: processing prompts [{shard_start}, {shard_end}) "
                        f"({shard_end - shard_start} of {len(prompts)} total)")
            prompts = prompts[shard_start:shard_end]
            raw_prompts = raw_prompts[shard_start:shard_end]
            output_path = shard_output_path(output_path, shard_id, num_shards)
            if not args.incremental_save:
                logger.info("Forcing --incremental_save for sharded execution")
                args.incremental_save = True
            if not prompts:
                logger.info(f"Shard {shard_id}: no prompts assigned, exiting")
                save_responses([], output_path)
                return

    logger.info(f"Running MISVO: model={model_name}, dataset={dataset_name}, "
                 f"n_samples={n_samples}, n_steps={n_steps}, lr={lr}, "
                 f"lam={lam}, regularizer={regularizer}, "
                 f"temperature={temperature}, top_p={top_p}, "
                 f"reward_norm={reward_norm_mode}")
    logger.info(f"Total generations per prompt: {n_samples * n_steps + 2}")

    results = try_resume(output_path) if args.incremental_save else []
    start_idx = len(results)

    # If vectors are requested but missing/incomplete, rewind start_idx
    vectors_path = output_path.replace(".json", "_vectors.pt")
    if args.save_vectors and start_idx > 0:
        n_existing_vecs = 0
        if os.path.exists(vectors_path):
            existing_vecs = torch.load(vectors_path, map_location="cpu", weights_only=True)
            n_existing_vecs = len(existing_vecs)
        if n_existing_vecs < start_idx:
            logger.info(f"Vectors incomplete ({n_existing_vecs}/{start_idx}), "
                        f"rewinding start_idx from {start_idx} to {n_existing_vecs}")
            results = results[:n_existing_vecs]
            start_idx = n_existing_vecs

    if start_idx > 0:
        logger.info(f"Skipping {start_idx} already-completed prompts")

    # --- Multi-GPU path. Each worker loads its own LLM + scorer on its
    # assigned GPU. For verifiable scorers the scorer is rule-based and
    # ~free to replicate; for RM-based scoring (SHP) the worker loads the
    # RM on its GPU alongside the LLM (assumes RM fits — small RMs only).
    # Route through the (multi-GPU) worker path whenever wave batching is
    # requested — the single-GPU path below has no wave support, only the
    # worker calls run_misvo_wave. num_gpus=1 spawns one worker on cuda:0.
    if args.num_gpus > 1 or wave_size > 1:
        logger.info(f"Worker mode: {args.num_gpus} GPU(s), wave_size={wave_size} "
                    f"(scoring={scoring_cfg.get('type', 'rm')})")
        from src.multi_gpu import run_multi_gpu
        vectors_path_arg = vectors_path if args.save_vectors else None
        results = run_multi_gpu(
            num_gpus=args.num_gpus,
            worker_fn=_misvo_worker,
            prompts=prompts,
            start_idx=start_idx,
            existing_results=results,
            output_path=output_path,
            worker_kwargs=dict(
                cfg=cfg, use_float_rewards=use_float_rewards,
                model_name=model_name, max_new_tokens=max_new_tokens,
                n_samples=n_samples, n_steps=n_steps, lr=lr, lam=lam,
                temperature=temperature, top_p=top_p,
                regularizer=regularizer,
                reward_norm_mode=reward_norm_mode,
                reward_norm_momentum=reward_norm_momentum, seed=args.seed,
                save_vectors=args.save_vectors,
                attn_implementation=attn_implementation,
                n_chunks=args.n_chunks,
                gen_chunks=args.gen_chunks,
                output_path=output_path,
                wave_size=wave_size,
                progress_desc="MISVO",
            ),
            vectors_path=vectors_path_arg,
        )
        _merge_worker_errors(output_path, args.num_gpus)
        # Strip generations from multi-GPU results and save separately
        gen_path = output_path.replace(".json", "_generations.json")
        multi_gens = [r.pop("generations_per_step", []) for r in results]
        import json as _json
        with open(gen_path, "w") as f:
            _json.dump(multi_gens, f, ensure_ascii=False)
        logger.info(f"Saved generations to {gen_path}")
        # Re-save results without generations
        from src.utils import save_responses as _save
        _save(results, output_path)
        logger.info(f"Saved {len(results)} results to {output_path}")
        import numpy as np
        rewards = [r["reward"] for r in results]
        logger.info(f"Mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    else:
        # --- Single-GPU path ---
        llm = BaseLLM(model_name,
                      attn_implementation=attn_implementation)
        llm.enable_thinking = cfg.get("generation", {}).get("enable_thinking")
        llm.load()

        if scoring_cfg.get("type") == "mbpp_plus":
            from src.evaluation.mbppplus_scorer import MBPPPlusScorer
            scorer = MBPPPlusScorer(timeout=scoring_cfg.get("timeout", 5))
            scorer.load()
        else:
            use_float_rewards = False
            scorer = create_reward_model(cfg["reward_model"]["id"], devices=args.rm_device, dtype=torch.bfloat16)
            scorer.load()

        if use_float_rewards:
            logger.info("Using float rewards for optimization (binary for ground truth)")

            def reward_fn(prompt_text, response_text):
                return scorer.score_float(prompt_text, response_text)

            def reward_fn_batch(prompt_text, responses):
                return scorer.score_responses_float(
                    [prompt_text] * len(responses), responses)
        else:
            def reward_fn(prompt_text, response_text):
                return scorer.score(prompt_text, response_text)

            def reward_fn_batch(prompt_text, responses):
                return scorer.score_responses(
                    [prompt_text] * len(responses), responses, batch_size=args.batch_size
                )

        steering_vectors = []
        if args.save_vectors and start_idx > 0 and os.path.exists(vectors_path):
            steering_vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)
            logger.info(f"Resumed {len(steering_vectors)} steering vectors")

        all_generations = []  # accumulated separately from main results
        errors = _load_existing_errors(output_path)

        for idx, prompt in enumerate(tqdm(prompts[start_idx:], desc="MISVO", initial=start_idx, total=len(prompts)), start=start_idx):
            prompt_text = get_prompt_text(prompt)
            logger.info(f"Prompt {idx+1}/{len(prompts)}")
            prompt_ids, attn_mask = llm.encode_prompt(prompt)

            # Binary scorer for per-iter tracking (when optimizing with float rewards)
            # (the single-GPU path never enabled it; the worker path does)
            binary_reward_fn_for_tracking = None

            try:
                result = run_misvo(
                    llm=llm,
                    prompt_ids=prompt_ids,
                    attention_mask=attn_mask,
                    reward_fn=reward_fn,
                    prompt_text=prompt_text,
                    lam=lam,
                    n_samples=n_samples,
                    n_steps=n_steps,
                    lr=lr,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    reward_fn_batch=reward_fn_batch,
                    reward_normalizer=reward_normalizer,
                    regularizer=regularizer,
                    binary_reward_fn=binary_reward_fn_for_tracking,
                    n_chunks=args.n_chunks,
                    gen_chunks=args.gen_chunks,
                )
            except Exception as e:
                logger.error(f"Prompt {idx+1}/{len(prompts)} FAILED, skipping: "
                             f"{type(e).__name__}: {e}")
                errors.append(_make_error_entry(idx, prompt_text, e))
                _save_errors(output_path, errors)
                torch.cuda.empty_cache()
                free_memory()
                continue

            entry = {
                "prompt": prompt_text,
                "response": result["best_response"],
                "mean_per_iter": result["mean_per_iter"],
                "best_per_iter": result["best_per_iter"],
                "final_reward": result["final_reward"],
                "fisher_per_iter": result["fisher_per_iter"],
                "fully_adaptive_fisher_per_iter": result["fully_adaptive_fisher_per_iter"],
                "kl_per_iter": result["kl_per_iter"],
                "l2_per_iter": result["l2_per_iter"],
                "u_norm_per_iter": result.get("u_norm_per_iter"),
            }
            if "binary_mean_per_iter" in result:
                entry["binary_mean_per_iter"] = result["binary_mean_per_iter"]
                entry["binary_best_per_iter"] = result["binary_best_per_iter"]
                entry["binary_at_float_best_per_iter"] = result.get("binary_at_float_best_per_iter")
            if use_float_rewards:
                # Binary reward = best binary found (oracle), not binary of float-best
                entry["reward"] = result.get("best_binary_reward",
                                             scorer.score(prompt_text, result["best_response"]))
                entry["float_reward"] = result["best_reward"]
                entry["response"] = result.get("best_binary_response", result["best_response"])
            else:
                entry["reward"] = result["best_reward"]
            # Pop generations to keep main JSON clean; save separately
            all_generations.append(entry.pop("generations_per_step", []))
            results.append(entry)
            if args.save_vectors:
                steering_vectors.append(result["steering_vector"])
            if args.incremental_save:
                incremental_save(results, output_path)
                gen_path = output_path.replace(".json", "_generations.json")
                with open(gen_path, "w") as f:
                    json.dump(all_generations, f, ensure_ascii=False)
                if args.save_vectors:
                    torch.save(steering_vectors, vectors_path)
            free_memory()

        if args.save_vectors and steering_vectors:
            torch.save(steering_vectors, vectors_path)
            logger.info(f"Saved {len(steering_vectors)} steering vectors to {vectors_path}")

        llm.unload()
        scorer.unload()

        # Save generations to separate file
        gen_path = output_path.replace(".json", "_generations.json")
        with open(gen_path, "w") as f:
            json.dump(all_generations, f, ensure_ascii=False)
        logger.info(f"Saved generations to {gen_path}")
        save_responses(results, output_path)
        logger.info(f"Saved {len(results)} results to {output_path}")
        _save_errors(output_path, errors)
        if errors:
            logger.warning(f"{len(errors)} prompts failed; see {_errors_path(output_path)}")

        # Print summary
        import numpy as np
        rewards = [r["reward"] for r in results]
        logger.info(f"Mean reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
