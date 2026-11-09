"""
Multi-GPU prompt distribution for IFBench experiments.

Spawns one worker process per GPU, each loading its own LLM + IFBench scorer.
Prompts are distributed dynamically via a shared work queue — each worker pulls
the next available prompt index when it finishes the current one, so no GPU
sits idle while others have remaining work.

Results are collected via a result Queue and saved incrementally as a contiguous
prefix (prompt i is never saved unless all prompts < i are also complete).

Steering vectors are saved to per-prompt files by each worker (bypassing the
queue to avoid tensor serialization issues with spawn), then merged by the
main process when the watermark advances.

Only used for IFBench scoring (CPU-only scorer). When using a GPU reward model,
multi-GPU complicates device management, so single-GPU mode is used.
"""

import os
import queue
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from src.utils import incremental_save, incremental_save_vectors, save_responses, logger


def _per_prompt_vector_path(vectors_dir, idx):
    """Path for a single prompt's steering vector file."""
    return os.path.join(vectors_dir, f"v_{idx:05d}.pt")


def _check_workers_alive(processes):
    """Check if all worker processes are still running.

    Returns:
        (all_alive, failed_pids) where failed_pids is list of (gpu_id, exitcode) for dead workers
    """
    failed = []
    for gpu_id, p in enumerate(processes):
        if not p.is_alive() and p.exitcode is not None and p.exitcode != 0:
            failed.append((gpu_id, p.exitcode))
    return len(failed) == 0, failed


def _terminate_workers(processes):
    """Terminate all worker processes that are still alive."""
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)


def _collect_results(result_queue, results, start_idx, total_remaining,
                     output_path, save_watermark, total_prompts, desc="Prompts",
                     vectors=None, vectors_path=None, vectors_dir=None,
                     processes=None):
    """Collect results from workers, save contiguous prefix incrementally.

    Monitors worker processes for crashes. If any worker dies (e.g. OOM),
    terminates all remaining workers and saves whatever results were collected.

    Args:
        result_queue: mp.Queue yielding (prompt_idx, result_entry) tuples
        results: pre-allocated list[None | dict] of length N
        start_idx: first prompt index that needs processing
        total_remaining: number of prompts still to process
        output_path: path for incremental saves
        save_watermark: first unsaved contiguous index
        total_prompts: total number of prompts (for progress bar)
        desc: description for the progress bar
        vectors: pre-allocated list[None | Tensor] for steering vectors (optional)
        vectors_path: path for merged vectors file (optional)
        vectors_dir: directory where workers save per-prompt vector files (optional)
        processes: list of mp.Process objects to monitor for crashes

    Returns:
        (updated save_watermark, worker_failed: bool)
    """
    completed = 0
    worker_failed = False
    skipped_indices = set()
    pbar = tqdm(total=total_prompts, initial=start_idx, desc=desc)
    while completed < total_remaining:
        # Use timeout to periodically check worker health
        try:
            idx, entry = result_queue.get(timeout=5)
        except queue.Empty:
            # No result yet — check if workers are still alive
            if processes is not None:
                all_alive, failed = _check_workers_alive(processes)
                if not all_alive:
                    for gpu_id, exitcode in failed:
                        logger.error(f"Worker on GPU {gpu_id} died with exit code {exitcode}")
                    # Check if all workers are dead (normal completion vs crash)
                    all_done = all(not p.is_alive() for p in processes)
                    if all_done and completed >= total_remaining:
                        break
                    if not all_alive and not all_done:
                        logger.error("Terminating remaining workers and saving partial results...")
                        _terminate_workers(processes)
                        # Drain any remaining results from the queue
                        while True:
                            try:
                                idx, entry = result_queue.get_nowait()
                                if entry is not None:
                                    results[idx] = entry
                                completed += 1
                                pbar.update(1)
                            except queue.Empty:
                                break
                        worker_failed = True
                        break
            continue

        if entry is not None:
            results[idx] = entry
        else:
            logger.warning(f"Prompt {idx} skipped by worker (OOM or error)")
            skipped_indices.add(idx)
        completed += 1
        pbar.update(1)

        # Load steering vector from per-prompt file written by the worker
        if vectors is not None and vectors_dir is not None:
            vpath = _per_prompt_vector_path(vectors_dir, idx)
            if os.path.exists(vpath):
                vectors[idx] = torch.load(vpath, map_location="cpu", weights_only=True)

        # Advance watermark over contiguous completed results (filled or skipped)
        old_watermark = save_watermark
        while save_watermark < len(results) and (
            results[save_watermark] is not None or save_watermark in skipped_indices
        ):
            save_watermark += 1

        if save_watermark > old_watermark:
            contiguous = [r for r in results[:save_watermark] if r is not None]
            incremental_save(contiguous, output_path)
            # Save merged vectors incrementally
            if vectors is not None and vectors_path is not None:
                contiguous_vecs = [v for v in vectors[:save_watermark] if v is not None]
                if contiguous_vecs:
                    incremental_save_vectors(contiguous_vecs, vectors_path)
            pbar.set_postfix(saved=f"{save_watermark}/{total_prompts}")
        else:
            pbar.set_postfix(saved=f"{old_watermark}/{total_prompts}",
                             waiting=f"p{idx} done, need p{old_watermark}")

    pbar.close()
    return save_watermark, worker_failed


def run_multi_gpu(num_gpus, worker_fn, prompts, start_idx, existing_results,
                  output_path, worker_kwargs, vectors_path=None):
    """Distribute prompts across GPUs dynamically and collect results.

    Uses a shared work queue so that faster GPUs automatically pick up more
    prompts — no GPU sits idle while others still have work.

    Args:
        num_gpus: number of GPUs to use
        worker_fn: callable(gpu_id, work_queue, prompts, result_queue, **worker_kwargs)
                   Must be a top-level function (picklable for mp.spawn).
                   The worker should loop pulling indices from work_queue until empty.
        prompts: full list of formatted prompt strings
        start_idx: index to resume from (len(existing_results))
        existing_results: list of already-completed result dicts (from resume)
        output_path: path for saving results JSON
        worker_kwargs: dict of additional kwargs passed to worker_fn
        vectors_path: if set, extract and save steering vectors incrementally

    Returns:
        list of all result dicts in prompt order
    """
    mp.set_start_method("spawn", force=True)

    desc = worker_kwargs.pop("progress_desc", "Prompts")
    N = len(prompts)
    # Pre-allocate with existing results
    results = [None] * N
    for i, r in enumerate(existing_results):
        results[i] = r

    # Set up vectors directory for file-based vector transfer
    vectors = None
    vectors_dir = None
    if vectors_path is not None:
        vectors = [None] * N
        vectors_dir = vectors_path + ".parts"
        os.makedirs(vectors_dir, exist_ok=True)
        # Load existing merged vectors on resume
        if start_idx > 0 and os.path.exists(vectors_path):
            existing_vecs = torch.load(vectors_path, map_location="cpu", weights_only=True)
            for i, v in enumerate(existing_vecs):
                vectors[i] = v
            logger.info(f"Resumed {len(existing_vecs)} steering vectors")
        # Pass vectors_dir to workers so they can save per-prompt files
        worker_kwargs["vectors_dir"] = vectors_dir

    # Build shared work queue with remaining prompt indices
    remaining = list(range(start_idx, N))
    if not remaining:
        logger.info("All prompts already completed, nothing to do")
        return existing_results

    work_queue = mp.Queue()
    for idx in remaining:
        work_queue.put(idx)

    logger.info(f"Work queue: {len(remaining)} prompts across {num_gpus} GPUs (dynamic scheduling)")

    result_queue = mp.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, work_queue, prompts, result_queue),
            kwargs=worker_kwargs,
        )
        p.start()
        processes.append(p)

    # Collect all results with contiguous-prefix saving
    save_watermark = start_idx
    save_watermark, worker_failed = _collect_results(
        result_queue, results, start_idx, len(remaining),
        output_path, save_watermark, total_prompts=N, desc=desc,
        vectors=vectors, vectors_path=vectors_path, vectors_dir=vectors_dir,
        processes=processes,
    )

    for p in processes:
        p.join(timeout=30)

    # Final save of complete results (partial if a worker failed)
    all_results = [r for r in results if r is not None]
    save_responses(all_results, output_path)
    if worker_failed:
        logger.error(f"Saved {len(all_results)}/{N} results (partial — worker crashed)")

    # Final save of all vectors + cleanup
    if vectors is not None and vectors_path is not None:
        all_vecs = [v for v in vectors if v is not None]
        if all_vecs:
            torch.save(all_vecs, vectors_path)
            logger.info(f"Saved {len(all_vecs)} steering vectors to {vectors_path}")
        # Clean up per-prompt files
        if vectors_dir is not None:
            import shutil
            try:
                shutil.rmtree(vectors_dir)
            except OSError:
                pass

    if worker_failed:
        raise RuntimeError(
            f"One or more GPU workers crashed. Saved {len(all_results)}/{N} partial results to {output_path}. "
            f"Re-run with --incremental_save to resume from where it left off."
        )

    return all_results
