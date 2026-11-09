"""
Merge shard / range result files into a single output JSON.

Two modes:
  - shards: equal-slice splits with `_shard{i}of{N}.json` naming
            (the SLURM-array workflow).
  - ranges: explicit prompt-index windows with `_range{S}-{E}.json`
            naming (resume-on-extra-nodes workflow). Ranges are merged
            into the BASE file in-place and must be contiguous from the
            base file's existing length to the end.

Usage:
    # Shard mode (existing behavior)
    python scripts/merge_shards.py --base_path results/dir/aisp_..._seed42.json --num_shards 4
    python scripts/merge_shards.py --results_dir results/dir/ --num_shards 4
    python scripts/merge_shards.py --base_path ... --num_shards 4 --merge_vectors --cleanup

    # Range mode — merges <base>.json + every <base>_range{S}-{E}.json sibling
    python scripts/merge_shards.py --mode ranges --base_path results/dir/misvo_....json
    python scripts/merge_shards.py --mode ranges --base_path ... --cleanup
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import save_responses, logger, setup_logging


def discover_base_paths(results_dir, num_shards):
    """Find all base paths that have shard0 files in the directory."""
    pattern = os.path.join(results_dir, f"*_shard0of{num_shards}.json")
    shard0_files = sorted(glob.glob(pattern))
    base_paths = []
    for path in shard0_files:
        # Strip _shard0of{N} to get the base path
        base = re.sub(rf"_shard0of{num_shards}\.json$", ".json", path)
        base_paths.append(base)
    return base_paths


def merge_json_shards(base_path, num_shards):
    """Merge JSON shard files into a single list. Returns (merged_list, missing_shards)."""
    root, ext = os.path.splitext(base_path)
    all_results = []
    missing = []
    for i in range(num_shards):
        shard_path = f"{root}_shard{i}of{num_shards}{ext}"
        if not os.path.exists(shard_path):
            missing.append(i)
            continue
        with open(shard_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning(f"Shard {i}: expected list, got {type(data).__name__}")
            missing.append(i)
            continue
        logger.info(f"  Shard {i}: {len(data)} entries from {os.path.basename(shard_path)}")
        all_results.extend(data)
    return all_results, missing


def merge_vector_shards(base_path, num_shards):
    """Merge PyTorch vector shard files. Returns (merged_list, count)."""
    import torch
    root, _ = os.path.splitext(base_path)
    all_vectors = []
    for i in range(num_shards):
        vec_path = f"{root}_shard{i}of{num_shards}_vectors.pt"
        if os.path.exists(vec_path):
            vecs = torch.load(vec_path, map_location="cpu", weights_only=True)
            logger.info(f"  Shard {i} vectors: {len(vecs)} entries")
            all_vectors.extend(vecs)
    return all_vectors, len(all_vectors)


def cleanup_shards(base_path, num_shards, include_vectors=False):
    """Delete shard files after successful merge."""
    root, ext = os.path.splitext(base_path)
    removed = 0
    for i in range(num_shards):
        shard_path = f"{root}_shard{i}of{num_shards}{ext}"
        if os.path.exists(shard_path):
            os.remove(shard_path)
            removed += 1
        # Also clean up candidates shards (for BoN)
        cand_path = f"{root}_shard{i}of{num_shards}_candidates.json"
        if os.path.exists(cand_path):
            os.remove(cand_path)
            removed += 1
        if include_vectors:
            vec_path = f"{root}_shard{i}of{num_shards}_vectors.pt"
            if os.path.exists(vec_path):
                os.remove(vec_path)
                removed += 1
    return removed


def merge_one(base_path, num_shards, merge_vectors=False, expected_total=None, cleanup=False):
    """Merge all shards for a single base path. Returns True on success."""
    logger.info(f"Merging: {os.path.basename(base_path)}")

    results, missing = merge_json_shards(base_path, num_shards)
    if missing:
        logger.error(f"  Missing shards: {missing} — skipping merge")
        return False

    if expected_total and len(results) != expected_total:
        logger.warning(f"  Expected {expected_total} results, got {len(results)}")

    save_responses(results, base_path)
    logger.info(f"  Merged {len(results)} results -> {base_path}")

    if merge_vectors:
        import torch
        vectors, count = merge_vector_shards(base_path, num_shards)
        if count > 0:
            root, _ = os.path.splitext(base_path)
            vec_path = f"{root}_vectors.pt"
            torch.save(vectors, vec_path)
            logger.info(f"  Merged {count} vectors -> {vec_path}")

    if cleanup:
        removed = cleanup_shards(base_path, num_shards, include_vectors=merge_vectors)
        logger.info(f"  Cleaned up {removed} shard files")

    return True


def discover_range_files(base_path):
    """Find all `<base>_range{S}-{E}.json` siblings of base_path. Returns list of
    (start, end, path) sorted by start."""
    root, ext = os.path.splitext(base_path)
    pattern = f"{root}_range*-*{ext}"
    found = []
    range_re = re.compile(rf"^{re.escape(root)}_range(\d+)-(\d+){re.escape(ext)}$")
    for path in glob.glob(pattern):
        m = range_re.match(path)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        found.append((s, e, path))
    found.sort()
    return found


def merge_ranges(base_path, cleanup=False):
    """Merge `<base>_range*-*.json` siblings into base_path. Returns True on success.

    Validates that ranges are contiguous starting from the existing base file's
    length (so a partial non-sharded run + range files = a complete output).
    """
    logger.info(f"Range-merging into: {os.path.basename(base_path)}")
    base_results = []
    if os.path.exists(base_path):
        with open(base_path) as f:
            base_results = json.load(f)
        logger.info(f"  Base file: {len(base_results)} entries")
    else:
        logger.info(f"  Base file does not exist; merging ranges only")

    range_files = discover_range_files(base_path)
    if not range_files:
        logger.warning(f"  No range files found alongside {base_path}")
        return False

    cursor = len(base_results)
    merged = list(base_results)
    used_paths = []
    for start, end, path in range_files:
        if start != cursor:
            logger.error(f"  Range file {os.path.basename(path)} starts at {start}, "
                         f"expected {cursor} (gap or overlap). Aborting.")
            return False
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"  {os.path.basename(path)}: expected list, got {type(data).__name__}")
            return False
        if len(data) != end - start:
            logger.warning(f"  {os.path.basename(path)}: expected {end - start} entries, "
                           f"got {len(data)} (range may be incomplete)")
        logger.info(f"  Range [{start}, {end}): {len(data)} entries from {os.path.basename(path)}")
        merged.extend(data)
        cursor = start + len(data)
        used_paths.append(path)

    save_responses(merged, base_path)
    logger.info(f"  Merged {len(merged)} results -> {base_path}")

    if cleanup:
        removed = 0
        for p in used_paths:
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
            # Also clean up matching errors / generations / vectors files
            root, ext = os.path.splitext(p)
            for suffix in ("_errors.json", "_generations.json", "_vectors.pt"):
                companion = root + suffix
                if os.path.exists(companion):
                    try:
                        os.remove(companion)
                        removed += 1
                    except OSError:
                        pass
        logger.info(f"  Cleaned up {removed} range files")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge shard / range result files")
    parser.add_argument("--mode", type=str, default="shards", choices=["shards", "ranges"],
                        help="shards = equal-slice (_shard{i}of{N}); ranges = explicit windows (_range{S}-{E})")
    parser.add_argument("--base_path", type=str, default=None,
                        help="Base output path (without partition suffix)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to scan (shards mode only — auto-discovers all runs)")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards (required for shards mode)")
    parser.add_argument("--expected_total", type=int, default=None,
                        help="Expected total number of results (for validation)")
    parser.add_argument("--merge_vectors", action="store_true", default=False,
                        help="Also merge steering vector .pt files (shards mode only)")
    parser.add_argument("--cleanup", action="store_true", default=False,
                        help="Delete partition files after successful merge")
    args = parser.parse_args()

    setup_logging()

    if args.mode == "ranges":
        if not args.base_path:
            parser.error("--base_path is required for --mode ranges")
        ok = merge_ranges(args.base_path, cleanup=args.cleanup)
        logger.info("Done." if ok else "Failed.")
        return

    # shards mode (default)
    if args.num_shards is None:
        parser.error("--num_shards is required for --mode shards")
    if not args.base_path and not args.results_dir:
        parser.error("Either --base_path or --results_dir is required")

    if args.base_path:
        base_paths = [args.base_path]
    else:
        base_paths = discover_base_paths(args.results_dir, args.num_shards)
        if not base_paths:
            logger.info(f"No shard files found in {args.results_dir} for num_shards={args.num_shards}")
            return
        logger.info(f"Found {len(base_paths)} sharded runs to merge")

    success = 0
    for base_path in base_paths:
        if merge_one(base_path, args.num_shards,
                     merge_vectors=args.merge_vectors,
                     expected_total=args.expected_total,
                     cleanup=args.cleanup):
            success += 1

    logger.info(f"Done: {success}/{len(base_paths)} merged successfully")


if __name__ == "__main__":
    main()
