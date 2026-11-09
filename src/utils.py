import torch
import random
import numpy as np
import logging
import os
import json

logger = logging.getLogger("aisp")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_responses(responses: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)


def incremental_save(responses: list, path: str):
    """Atomically save results to prevent corruption from interrupted writes.

    Writes to a temporary file, fsyncs to guarantee data hits persistent storage,
    then renames (atomic on POSIX). Safe on networked filesystems (NFS, Lustre).
    """
    import tempfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        # fsync the directory to ensure the rename is persisted
        dir_fd = os.open(dir_name, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_generations(results: list, output_path: str):
    """Extract generations_per_step from results and save to a separate file.

    Modifies results in-place (removes generations_per_step from each entry).
    Saves to {output_path}_generations.json.
    """
    gen_path = output_path.replace(".json", "_generations.json")
    generations = []
    for entry in results:
        generations.append(entry.pop("generations_per_step", []))
    os.makedirs(os.path.dirname(gen_path) or ".", exist_ok=True)
    with open(gen_path, "w") as f:
        json.dump(generations, f, ensure_ascii=False)
    logger.info(f"Saved generations to {gen_path}")


def try_resume(path: str) -> list:
    """Load existing results for resuming, or return empty list if no file exists."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"Resuming: found {len(data)} existing results in {path}")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not resume from {path}: {e}, starting fresh")
    return []


def incremental_save_vectors(vectors: list, path: str):
    """Atomically save steering vectors list to a .pt file.

    Same atomic write pattern as incremental_save but for torch tensors.
    """
    import tempfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp.pt")
    try:
        os.close(fd)
        torch.save(vectors, tmp_path)
        os.replace(tmp_path, path)
        dir_fd = os.open(dir_name, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Sharding helpers for SLURM job array support
# ---------------------------------------------------------------------------

def add_shard_args(parser):
    """Add --shard_id / --num_shards / --start_idx / --end_idx arguments.

    Two distinct partition modes are supported:
      - shard mode: --shard_id / --num_shards (equal-slice split, e.g. SLURM arrays)
      - range mode: --start_idx / --end_idx (explicit prompt-index window)
    The two modes are mutually exclusive. Range mode is useful for resuming
    a partial non-sharded run on additional nodes (start_idx = len(existing
    main output.json), end_idx = total) without redoing equal-slice math.
    """
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard index (0-based). Auto-detected from SLURM_ARRAY_TASK_ID if not given.")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards. Auto-detected from SLURM_ARRAY_TASK_COUNT if not given.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Range mode: first prompt index to process (inclusive). "
                             "Use with --end_idx. Mutually exclusive with --shard_id/--num_shards.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Range mode: prompt index to stop before (exclusive). "
                             "Use with --start_idx.")
    return parser


def resolve_shard_args(args):
    """Resolve shard_id from SLURM env if not given; validate shard arguments.

    Returns (shard_id, num_shards) or (None, None) if sharding is not active.

    Returns (None, None) if range mode (--start_idx/--end_idx) is in use,
    even if SLURM array env vars are set — explicit range overrides arrays.
    """
    if getattr(args, "start_idx", None) is not None or getattr(args, "end_idx", None) is not None:
        return None, None

    if args.shard_id is None and args.num_shards is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        num_tasks = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if task_id is not None and num_tasks is not None:
            args.shard_id = int(task_id)
            args.num_shards = int(num_tasks)
            logger.info(f"Auto-detected SLURM array: shard_id={args.shard_id}, num_shards={args.num_shards}")
        else:
            return None, None

    if args.shard_id is not None and args.num_shards is None:
        raise ValueError("--num_shards is required when --shard_id is set")
    if args.num_shards is not None and args.shard_id is None:
        raise ValueError("--shard_id is required when --num_shards is set")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id={args.shard_id} out of range [0, {args.num_shards})")
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        raise ValueError("--num_gpus > 1 and --shard_id are mutually exclusive. "
                         "Use one GPU per shard with job arrays.")
    return args.shard_id, args.num_shards


def resolve_range_args(args, total):
    """Validate --start_idx / --end_idx and return (start, end) or (None, None).

    Range mode and shard mode are mutually exclusive — caller should treat
    a non-None return here as an override of any shard arguments.

    Args:
        args: argparse namespace with start_idx/end_idx attributes
        total: total number of prompts (for clamping/validation)
    """
    s = getattr(args, "start_idx", None)
    e = getattr(args, "end_idx", None)
    if s is None and e is None:
        return None, None
    if s is None or e is None:
        raise ValueError("--start_idx and --end_idx must both be set together")
    if (args.shard_id is not None) or (args.num_shards is not None):
        raise ValueError("--start_idx/--end_idx and --shard_id/--num_shards are mutually exclusive")
    if s < 0:
        raise ValueError(f"--start_idx={s} must be >= 0")
    if e <= s:
        raise ValueError(f"--end_idx={e} must be > --start_idx={s}")
    if e > total:
        logger.warning(f"--end_idx={e} clamped to total prompts={total}")
        e = total
    # Range mode is compatible with --num_gpus > 1: range slices the prompt
    # list, then run_multi_gpu distributes that slice across local GPUs.
    return s, e


def range_output_path(base_path, start, end):
    """Convert 'foo.json' to 'foo_range{S}-{E}.json' (works for any extension)."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_range{start}-{end}{ext}"


def compute_shard_slice(total, shard_id, num_shards):
    """Compute (start, end) indices for a contiguous shard of `total` items.

    Distributes remainder to the first `total % num_shards` shards so sizes
    differ by at most 1.
    """
    base_size = total // num_shards
    remainder = total % num_shards
    if shard_id < remainder:
        start = shard_id * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (shard_id - remainder) * base_size
        end = start + base_size
    return start, end


def shard_output_path(base_path, shard_id, num_shards):
    """Convert 'foo.json' to 'foo_shard0of4.json' (works for any extension)."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_shard{shard_id}of{num_shards}{ext}"


def errors_path(output_path):
    """Path to the merged errors JSON file (sibling of output_path)."""
    return output_path.replace(".json", "_errors.json")


def per_worker_errors_path(output_path, gpu_id):
    """Per-worker temporary JSONL file for collecting errors."""
    return output_path.replace(".json", f"_errors_gpu{gpu_id}.tmp.jsonl")


def append_worker_error(output_path, gpu_id, idx, prompt_text, exc):
    """Append a single failed-prompt record to the per-worker JSONL."""
    if output_path is None:
        return
    entry = {
        "prompt_index": idx,
        "prompt_number": idx + 1,
        "prompt": prompt_text,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    with open(per_worker_errors_path(output_path, gpu_id), "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def merge_worker_errors(output_path, num_gpus):
    """Merge per-worker JSONL files into <output_path>_errors.json and clean up."""
    if output_path is None:
        return
    out_path = errors_path(output_path)
    errors = []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                errors = json.load(f)
        except (json.JSONDecodeError, OSError):
            errors = []
    seen = {(e.get("prompt_index"), e.get("error_message")) for e in errors}
    for gpu_id in range(num_gpus):
        p = per_worker_errors_path(output_path, gpu_id)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (e.get("prompt_index"), e.get("error_message"))
                if key in seen:
                    continue
                seen.add(key)
                errors.append(e)
        try:
            os.remove(p)
        except OSError:
            pass
    if errors:
        errors.sort(key=lambda e: e.get("prompt_index", -1))
        with open(out_path, "w") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        logger.warning(f"Saved {len(errors)} prompt errors to {out_path}")


def save_errors(output_path, errors):
    """Write errors list to <output_path>_errors.json (sorted by prompt index)."""
    if output_path is None or not errors:
        return
    errors_sorted = sorted(errors, key=lambda e: e.get("prompt_index", -1))
    with open(errors_path(output_path), "w") as f:
        json.dump(errors_sorted, f, indent=2, ensure_ascii=False)


def load_existing_errors(output_path):
    """Load the existing errors JSON if it exists, else return []."""
    if output_path is None:
        return []
    p = errors_path(output_path)
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def make_error_entry(idx, prompt_text, exc):
    """Build a single error record dict."""
    return {
        "prompt_index": idx,
        "prompt_number": idx + 1,
        "prompt": prompt_text,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
