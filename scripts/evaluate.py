"""
Evaluation script: computes reward, diversity, and coherence for results files
and draws the reward-trajectory (convergence) plots.

MBPP+ result files are re-scored once against the full MBPP+ (plus) test suite
and the plus-scored rewards are written back into the file.

Usage:
    python scripts/evaluate.py --results_dir results/lfm25_1b_shp
    python scripts/evaluate.py --file results/lfm25_1b_shp/misvo_..._seed42.json
    # paper-style trajectory figures, averaged over seeds:
    python scripts/evaluate.py --results_dir results/lfm25_1b_shp --paper_ready --average-runs
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.diversity import compute_diversity_batch
from src.evaluation.coherence import compute_coherence_batch
from src.utils import setup_logging, logger


# ---------------------------------------------------------------------------
# Result cache
#
# Per-directory cache that maps `(file_content_hash, num_prompts)` to the
# evaluate_file output. Stored as `<results_dir>/.evaluate_cache.json`. The
# cache key naturally distinguishes:
#   - the regular pass (num_prompts = args.num_prompts, often None)
#   - the --shared_prompts pass (num_prompts = shared_n, derived per run)
# so the same cache file serves both — no separate "shared" subkey needed.
#
# Bump CACHE_VERSION whenever evaluate_file's behaviour or return-dict schema
# changes — old entries with a different version are silently discarded.
# ---------------------------------------------------------------------------

CACHE_VERSION = 1
CACHE_FILENAME = ".evaluate_cache.json"


def _compute_file_hash(path: str) -> str:
    """SHA-256 hex digest of the file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(file_hash: str, num_prompts) -> str:
    """`num_prompts=None` (evaluate-all) is a distinct cache slot from any
    specific N value — encode that explicitly so it survives JSON round-trip."""
    n_tag = "all" if num_prompts is None else f"n{int(num_prompts)}"
    return f"{file_hash}_{n_tag}"


def _cache_path(results_dir: str) -> str:
    return os.path.join(results_dir, CACHE_FILENAME)


def _load_cache(results_dir: str) -> dict:
    """Returns {"version": int, "entries": {key: entry, ...}}. On missing
    file, version mismatch, or unparseable JSON, returns a fresh empty cache."""
    path = _cache_path(results_dir)
    if not os.path.exists(path):
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        with open(path, "r") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cache file {path} is unreadable ({e}); starting fresh.")
        return {"version": CACHE_VERSION, "entries": {}}
    if cache.get("version") != CACHE_VERSION:
        logger.info(f"Cache version mismatch (got {cache.get('version')}, "
                    f"expected {CACHE_VERSION}); discarding old cache.")
        return {"version": CACHE_VERSION, "entries": {}}
    if "entries" not in cache or not isinstance(cache["entries"], dict):
        return {"version": CACHE_VERSION, "entries": {}}
    return cache


def _save_cache(results_dir: str, cache: dict) -> None:
    """Atomic write to avoid leaving a half-written cache on crash."""
    path = _cache_path(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)


def _evaluate_file_cached(path: str, num_prompts, cache: dict,
                          use_cache: bool) -> dict:
    """Lookup-or-compute. On cache hit (matching file_hash + num_prompts),
    returns the stored metrics and logs the reuse. On miss, runs the real
    `evaluate_file` and writes a new entry into ``cache["entries"]`` (the
    caller is responsible for flushing the cache to disk afterwards)."""
    if use_cache:
        file_hash = _compute_file_hash(path)
        key = _cache_key(file_hash, num_prompts)
        entry = cache.get("entries", {}).get(key)
        if entry is not None and entry.get("file_hash") == file_hash:
            n_tag = "all" if num_prompts is None else f"n={num_prompts}"
            logger.info(f"  [cache hit, {n_tag}] {os.path.basename(path)}")
            return entry["metrics"]

    metrics = evaluate_file(path, num_prompts=num_prompts)

    if use_cache:
        cache.setdefault("entries", {})[key] = {
            "file_hash": file_hash,
            "file_path": path,
            "num_prompts": num_prompts,
            "metrics": metrics,
            "cached_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    return metrics


def _is_mbppplus_path(path: str) -> bool:
    p = path.lower()
    return "mbpp_plus" in p or "mbppplus" in p


def _ensure_mbppplus_plus_rescored(path: str, data: list) -> list:
    """For MBPP+ result files, ensure reward/float_reward come from the full
    augmented MBPP+ test suite — not the small `test_list` used at training time.

    Mirrors scripts/rescore_mbppplus.py: rescores each saved (prompt, response)
    via MBPPPlusScorer.score_plus_both and writes back, preserving the original
    training-time scores under reward_base/float_reward_base. Idempotent: skipped
    once reward_base is present.
    """
    if not data:
        return data
    if all("reward_base" in e for e in data):
        return data  # already plus-rescored

    from src.evaluation.mbppplus_scorer import MBPPPlusScorer
    logger.info(f"Rescoring {path} against full MBPP+ test suite (one-time, persisted)...")
    scorer = MBPPPlusScorer()
    scorer.load()
    try:
        for entry in data:
            if "reward_base" in entry:
                continue  # mixed file; leave already-rescored entries alone
            binary, fvalue = scorer.score_plus_both(entry["prompt"], entry["response"])
            entry["reward_base"] = entry.get("reward")
            entry["float_reward_base"] = entry.get("float_reward")
            entry["reward"] = binary
            entry["float_reward"] = fvalue
    finally:
        scorer.unload()

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  wrote plus-rescored rewards back to {path}")
    return data


def evaluate_file(path: str, num_prompts: int = None) -> dict:
    """Evaluate a single results file.

    Args:
        num_prompts: If set, only evaluate the first N entries.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if _is_mbppplus_path(path):
        data = _ensure_mbppplus_plus_rescored(path, data)

    if num_prompts is not None:
        data = data[:num_prompts]

    prompts = [d["prompt"] for d in data]
    responses = [d["response"] for d in data]
    rewards = [d.get("reward", None) for d in data]

    metrics = {"file": os.path.basename(path), "num_samples": len(data)}

    # Reward stats — recompute the best-so-far reward from the raw per-iteration
    # data when available (best over all K*N generations), rather than trusting
    # the "reward" field. MBPP+ files: skip the per-iter oracle (the training-time
    # tracker uses test_list, not the plus suite); use the plus-rescored
    # top-level reward instead.
    use_oracle_per_iter = not _is_mbppplus_path(path)
    recomputed_rewards = []
    for d in data:
        if not use_oracle_per_iter and d.get("reward") is not None:
            recomputed_rewards.append(d["reward"])
        else:
            if "binary_best_per_iter" in d and d["binary_best_per_iter"]:
                # AISP/MISVO with binary tracking: oracle binary best across all steps
                    recomputed_rewards.append(max(d["binary_best_per_iter"]))
            elif "best_per_iter" in d and d["best_per_iter"] and d.get("float_reward") is None:
                # AISP/MISVO without separate float tracking (SHP runs):
                # best_per_iter IS the primary reward.
                recomputed_rewards.append(max(d["best_per_iter"]))
            elif "all_rewards" in d:
                # BoN: oracle best across all candidates
                    recomputed_rewards.append(max(d["all_rewards"]))
            elif d.get("reward") is not None:
            # Fallback (and the path used for MBPP+ after plus-rescore)
                recomputed_rewards.append(d["reward"])

    if recomputed_rewards:
        metrics["reward_mean"] = float(np.mean(recomputed_rewards))
        metrics["reward_std"] = float(np.std(recomputed_rewards))
    else:
        metrics["reward_mean"] = None
        metrics["reward_std"] = None

    # Float reward stats (MBPP+ with float_rewards: fraction of tests passed)
    float_rewards = []
    for d in data:
        if d.get("float_reward") is not None:
            # Use best_per_iter if available and higher (fixes old buggy entries)
            fr = d["float_reward"]
            if "best_per_iter" in d and d["best_per_iter"]:
                fr = max(fr, max(d["best_per_iter"]))
            float_rewards.append(fr)
    if float_rewards:
        metrics["float_reward_mean"] = float(np.mean(float_rewards))
        metrics["float_reward_std"] = float(np.std(float_rewards))
    else:
        metrics["float_reward_mean"] = None
        metrics["float_reward_std"] = None

    # Diversity
    div_metrics = compute_diversity_batch(responses)
    metrics.update(div_metrics)

    # Coherence
    try:
        coh_metrics = compute_coherence_batch(prompts, responses)
        metrics.update(coh_metrics)
    except Exception as e:
        logger.warning(f"Coherence computation failed: {e}")
        metrics["coherence_mean"] = None
        metrics["coherence_std"] = None

    return metrics


# --- Display names for paper-ready legends --------------------------------
# Human-readable labels for the MISVO regularizer variants (paper notation).
MISVO_REG_DISPLAY = {
    "frozen_fisher": "F̄ (frozen reference)",
    "steered_traj_fisher": "F̄^u (on-policy rollouts)",
    "fully_adaptive_fisher": "F̃^u (on-policy Fisher)",
    "kl": "KL",
    "l2": "L2",
}

# Method display names for paper-ready legends. Falls back to method.upper() if absent.
METHOD_DISPLAY = {
    "aisp": "AISP",
    "misvo": "MISVO",
    "bon": "BoN",
}


def _extract_misvo_regularizer(filepath):
    """Extract the MISVO regularizer name from a filename.

    Filename pattern is `..._reg<regularizer>_<next_field>...`. We search for
    the known values directly to avoid ambiguity from underscores in names
    like 'frozen_fisher'.
    """
    base = os.path.basename(filepath).lower()
    for reg in ("fully_adaptive_fisher", "frozen_fisher",
                "steered_traj_fisher", "kl", "l2"):
        if f"_reg{reg}_" in base or base.endswith(f"_reg{reg}"):
            return reg
    return None


# Hyperparameters a paper legend may disambiguate on, in the order they are
# appended. Only those that actually *differ* between same-method runs are
# shown, so a plot with one run per method stays at bare "AISP" / "BoN".
# The MISVO regularizer is handled separately (see _paper_params).
PAPER_PARAM_PATTERNS = [
    ("λ", r'_lam([\d.]+)'),
    ("lr", r'_lr([\d.]+)'),
    ("n", r'_n(\d+)_k'),
    ("κ", r'_k(\d+)_'),
    ("N", r'_N(\d+)'),
    ("ns", r'_ns(\d+)'),
    ("steps", r'_steps(\d+)'),
    ("σ²", r'_sig([\d.]+)'),
    ("α", r'_a([\d.]+)(?:_|$)'),
]

# "reg" (MISVO regularizer) goes first so the legend reads "MISVO (Frozen)",
# matching how these variants have always been labelled.
PAPER_PARAM_ORDER = ["reg"] + [d for d, _ in PAPER_PARAM_PATTERNS]


def _paper_params(filepath):
    """Hyperparameter values parsed out of a filename, keyed by display name.

    The seed token is stripped first so seeds of the same scenario are
    indistinguishable here (they should share a legend entry).
    """
    base = re.sub(r'_s(?:eed)?\d+', '',
                  os.path.splitext(os.path.basename(filepath))[0])
    params = {}
    for display, pattern in PAPER_PARAM_PATTERNS:
        m = re.search(pattern, base)
        if m is not None:
            params[display] = m.group(1)
    reg = _extract_misvo_regularizer(filepath)
    if reg is not None:
        params["reg"] = reg
    return params


def _paper_label(filepath, all_filepaths):
    """Minimal legend label for paper-ready plots.

    The method's display name alone ("AISP", "MISVO", "BoN") whenever it is
    the only run of that method in the evaluation set. When several runs of
    one method are present, the hyperparameters that differ *between those
    runs* are appended — and only those — e.g. two AISP runs that vary only
    in lambda become "AISP (λ=0.1)" / "AISP (λ=1.0)", and MISVO variants
    become "MISVO (Frozen)" / "MISVO (Steered Trajectory)".
    """
    method = _detect_method(filepath)
    display = METHOD_DISPLAY.get(method, method.upper())

    siblings = [fp for fp in all_filepaths if _detect_method(fp) == method]
    if len(siblings) <= 1:
        return display

    params = _paper_params(filepath)
    sibling_params = [_paper_params(fp) for fp in siblings]
    quals = []
    for key in PAPER_PARAM_ORDER:
        if key not in params:
            continue
        if len({sp.get(key) for sp in sibling_params}) <= 1:
            continue  # shared by every run of this method — says nothing
        if key == "reg":
            quals.append(MISVO_REG_DISPLAY.get(params[key], params[key]))
        else:
            quals.append(f"{key}={params[key]}")

    return f"{display} ({', '.join(quals)})" if quals else display


def _detect_method(filename):
    """Infer method name from filename."""
    base = os.path.basename(filename).lower()
    for prefix in ["aisp", "misvo", "bon"]:
        if base.startswith(prefix):
            return prefix
    return "unknown"


def _make_label(filepath):
    """Create a short readable label from the filename.

    Extracts method, model, and distinguishing hyperparameters so that
    different runs of the same method are visually distinguishable.
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    import re
    base = re.sub(r'_s(?:eed)?\d+', '', base)
    method = _detect_method(filepath).upper()
    parts = [method]
    # Model name
    for model in ["llama3_8b_instruct", "gemma3_4b_it", "lfm25_1b", "phi4"]:
        if model in base.lower():
            parts.append(model.replace("_", "-"))
            break
    # Key hyperparams — extract all that appear in the filename
    param_patterns = [
        (r'_lam([\d.]+)', 'λ'),
        (r'_lr([\d.]+)', 'lr'),
        (r'_ns(\d+)', 'ns'),
        (r'_steps(\d+)', 'st'),
        (r'_n(\d+)_k(\d+)', None),  # handled separately
        (r'_reg([a-z0-9_]+?)_(?:norm|seed|floatrw)', 'reg'),
        (r'_norm([a-z]+)', 'norm'),
        (r'_N(\d+)', 'N'),
        (r'_sig([\d.]+)', 'σ²'),
        (r'_a([\d.]+)', 'α'),
    ]
    for pattern, label in param_patterns:
        m = re.search(pattern, base)
        if m is None:
            continue
        if label is None:
            # n + kappa pair
            parts.append(f"n={m.group(1)}")
            parts.append(f"k={m.group(2)}")
        elif label == 'norm' and m.group(1) == "none":
            continue
        else:
            parts.append(f"{label}={m.group(1)}")
    return " ".join(parts)


def _extract_convergence_curves(filepath, num_prompts=None):
    """Extract convergence data from a results file.

    For iterative methods (AISP, MISVO): uses mean_per_iter, best_per_iter.
    For BoN: computes running mean/best from all_rewards.

    Args:
        num_prompts: If set, only use the first N entries.

    Returns dict with:
        method, label, n_per_iter, kappa,
        mean_at_k (K,), best_at_k (K,), best_so_far (K,),
        x_samples (K,) — cumulative sample count at each iteration
    or None if no convergence data is available.
    """
    with open(filepath) as f:
        data = json.load(f)

    if num_prompts is not None:
        data = data[:num_prompts]

    method = _detect_method(filepath)
    label = _make_label(filepath)
    first = data[0]

    # --- Iterative methods: AISP, MISVO ---
    if "mean_per_iter" in first and "best_per_iter" in first:
        all_means = [d["mean_per_iter"] for d in data]
        all_bests = [d["best_per_iter"] for d in data]
        K = min(len(m) for m in all_means)
        all_means = np.array([m[:K] for m in all_means])
        all_bests = np.array([b[:K] for b in all_bests])

        avg_mean = all_means.mean(axis=0)
        avg_best = all_bests.mean(axis=0)
        avg_best_so_far = np.maximum.accumulate(all_bests, axis=1).mean(axis=0)

        # Infer samples per iteration from filename or method
        import re
        base = os.path.basename(filepath)
        if method == "aisp":
            m_match = re.search(r'_n(\d+)_', base)
            n_per_iter = int(m_match.group(1)) if m_match else 32
        elif method == "misvo":
            m_match = re.search(r'_ns(\d+)_', base)
            n_per_iter = int(m_match.group(1)) if m_match else 16
        else:
            n_per_iter = 32

        x_samples = np.arange(1, K + 1) * n_per_iter

        result = {
            "method": method, "label": label,
            "n_per_iter": n_per_iter, "kappa": K,
            "mean_at_k": avg_mean, "best_at_k": avg_best,
            "best_so_far": avg_best_so_far,
            "x_samples": x_samples,
        }

        # Binary reward curves (available with float_rewards, e.g. MBPP+)
        if "binary_mean_per_iter" in first and first["binary_mean_per_iter"] is not None:
            all_bin_means = [d["binary_mean_per_iter"] for d in data
                            if d.get("binary_mean_per_iter") is not None]
            all_bin_bests = [d["binary_best_per_iter"] for d in data
                            if d.get("binary_best_per_iter") is not None]
            if all_bin_means and all_bin_bests:
                K_bin = min(len(m) for m in all_bin_means)
                all_bin_means = np.array([m[:K_bin] for m in all_bin_means])
                all_bin_bests = np.array([b[:K_bin] for b in all_bin_bests])
                result["binary_mean_at_k"] = all_bin_means.mean(axis=0)
                result["binary_best_at_k"] = all_bin_bests.mean(axis=0)
                result["binary_best_so_far"] = np.maximum.accumulate(all_bin_bests, axis=1).mean(axis=0)

                # Float-selected pass@1 best-so-far: BINARY reward of the completion
                # with the highest FLOAT reward seen so far. This matches the reported
                # `reward` metric at the final budget (unlike binary_best_so_far, which
                # is the oracle binary max — a different, optimistic completion).
                if first.get("binary_at_float_best_per_iter") is not None:
                    coupled = [d["binary_at_float_best_per_iter"] for d in data
                               if d.get("binary_at_float_best_per_iter") is not None]
                    floats = [d["best_per_iter"] for d in data
                              if d.get("binary_at_float_best_per_iter") is not None]
                    if coupled:
                        Kc = min(min(len(c) for c in coupled), min(len(f) for f in floats))
                        cb = np.array([c[:Kc] for c in coupled], dtype=float)   # coupled binary
                        fb = np.array([f[:Kc] for f in floats], dtype=float)    # float best/iter
                        run_f = np.full(cb.shape[0], -np.inf)
                        run_b = cb[:, 0].copy()
                        sel = np.empty_like(cb)
                        for k in range(Kc):
                            upd = fb[:, k] >= run_f          # running float-argmax updated?
                            run_f = np.where(upd, fb[:, k], run_f)
                            run_b = np.where(upd, cb[:, k], run_b)
                            sel[:, k] = run_b
                        result["binary_pass1_best_so_far"] = sel.mean(axis=0)

        return result

    # --- BoN: compute convergence from all_rewards ---
    if "all_rewards" in first:
        # Use float rewards for the main convergence plot when available,
        # so the scale matches AISP/MISVO (which optimize float rewards).
        # Binary rewards are used for the separate binary convergence plot.
        if "all_float_rewards" in first:
            all_rewards_per_prompt = [d["all_float_rewards"] for d in data]
        else:
            all_rewards_per_prompt = [d["all_rewards"] for d in data]
        N = min(len(r) for r in all_rewards_per_prompt)
        all_rewards = np.array([r[:N] for r in all_rewards_per_prompt])  # (P, N)

        # Compute running mean and best-so-far at each sample count
        # Use the same x-axis granularity as AISP (every 32 samples)
        step = 32
        x_points = list(range(step, N + 1, step))
        if x_points[-1] != N:
            x_points.append(N)

        running_mean = np.zeros(len(x_points))
        running_best = np.zeros(len(x_points))
        for i, k in enumerate(x_points):
            running_mean[i] = all_rewards[:, :k].mean(axis=1).mean()
            running_best[i] = all_rewards[:, :k].max(axis=1).mean()

        result = {
            "method": method, "label": label,
            "n_per_iter": step, "kappa": len(x_points),
            "mean_at_k": running_mean,
            "best_at_k": running_best,
            "best_so_far": running_best,  # same for BoN (no iterations)
            "x_samples": np.array(x_points),
        }

        # Binary convergence curves for BoN (oracle binary best)
        all_binary_per_prompt = [d["all_rewards"] for d in data]
        N_bin = min(len(r) for r in all_binary_per_prompt)
        all_binary = np.array([r[:N_bin] for r in all_binary_per_prompt])
        bin_mean = np.zeros(len(x_points))
        bin_best = np.zeros(len(x_points))
        for i, k in enumerate(x_points):
            kk = min(k, N_bin)
            bin_mean[i] = all_binary[:, :kk].mean(axis=1).mean()
            bin_best[i] = all_binary[:, :kk].max(axis=1).mean()

        result["binary_mean_at_k"] = bin_mean
        result["binary_best_at_k"] = bin_best
        result["binary_best_so_far"] = bin_best  # already monotonically non-decreasing for BoN

        return result

    return None


# --- Multi-seed aggregation ------------------------------------------------
# Runs of the same scenario differ only by the seed token in their filename
# (`..._seed42.json` / `..._s42.json`). Stripping it yields a stable group key,
# and `_make_label` / `_paper_label` already ignore the seed, so grouped curves
# keep the same legend entry they had before.

# Series produced by _extract_convergence_curves that are averaged across seeds.
_CURVE_SERIES_KEYS = ("mean_at_k", "best_at_k", "best_so_far",
                      "binary_mean_at_k", "binary_best_at_k",
                      "binary_best_so_far", "binary_pass1_best_so_far")


def _seed_stripped_key(filepath):
    """Group key for multi-seed runs: basename with the seed token removed."""
    base = os.path.splitext(os.path.basename(filepath))[0]
    return re.sub(r'_s(?:eed)?\d+', '', base)


def _aggregate_curves_by_seed(curves, resolved_paths):
    """Collapse multi-seed runs of the same scenario into a single curve.

    Within a group, every series is truncated to the shortest member and
    averaged across seeds; the per-point (per-x) std across seeds is stored
    under ``"<key>_std"`` and the seed count under ``"n_runs"``. Single-run
    groups pass through untouched (no ``_std`` key → no shaded band).

    Returns (curves, paths) in first-appearance order, with one representative
    path per group so downstream label lookups still work.
    """
    groups, order = {}, []
    for curve, path in zip(curves, resolved_paths):
        key = _seed_stripped_key(path)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((curve, path))

    agg_curves, agg_paths = [], []
    for key in order:
        members = groups[key]
        if len(members) == 1:
            agg_curves.append(members[0][0])
            agg_paths.append(members[0][1])
            continue

        merged = dict(members[0][0])
        merged["n_runs"] = len(members)
        # Align on the shortest member so a seed that stopped early doesn't
        # distort the tail of the average.
        K = min(len(c["x_samples"]) for c, _ in members)
        merged["x_samples"] = np.asarray(members[0][0]["x_samples"])[:K]
        merged["kappa"] = K
        for sk in _CURVE_SERIES_KEYS:
            present = [c[sk] for c, _ in members if sk in c]
            if len(present) != len(members):
                # Some seed lacks this series (e.g. no binary tracking) —
                # drop it rather than averaging an inconsistent subset.
                merged.pop(sk, None)
                continue
            Ks = min(K, min(len(s) for s in present))
            arr = np.array([np.asarray(s)[:Ks] for s in present])  # (S, Ks)
            merged[sk] = arr.mean(axis=0)
            merged[sk + "_std"] = arr.std(axis=0)
        agg_curves.append(merged)
        agg_paths.append(members[0][1])

    n_grouped = sum(1 for c in agg_curves if c.get("n_runs", 1) > 1)
    if n_grouped:
        logger.info(f"--average_runs: collapsed {len(curves)} runs into "
                    f"{len(agg_curves)} curves ({n_grouped} multi-seed)")
    else:
        logger.info("--average_runs: no multi-seed groups found "
                    "(every scenario has a single run)")
    return agg_curves, agg_paths


def _annotate_seed_counts(curves):
    """Append the seed count to the legend label of averaged curves."""
    for curve in curves:
        if curve.get("n_runs", 1) > 1:
            curve["label"] = f"{curve['label']} ({curve['n_runs']} seeds)"


def _apply_paper_labels(curves, resolved_paths):
    """Assign minimal paper labels, warning when they hide a real difference.

    Different seeds of one scenario are *meant* to share a label. Two genuinely
    different scenarios sharing one is not: they'd be drawn in the same colour
    under a single legend entry. That means the filenames differ in a field
    `PAPER_PARAM_PATTERNS` doesn't cover, so say so rather than mislead.
    """
    by_label = {}
    for curve, path in zip(curves, resolved_paths):
        curve["label"] = _paper_label(path, resolved_paths)
        by_label.setdefault(curve["label"], []).append(path)

    for label, paths in by_label.items():
        scenarios = {_seed_stripped_key(p) for p in paths}
        if len(scenarios) > 1:
            logger.warning(
                f"--paper_ready: {len(scenarios)} different runs all label as "
                f"\"{label}\" and will share one legend entry: "
                + ", ".join(sorted(os.path.basename(p) for p in paths)))


def _curve_colors_and_legend(curves, plt, paper_ready=False):
    """Per-curve colours plus the label each curve contributes to the legend.

    Paper-ready plots colour by *label*, so curves that collapse to the same
    minimal name (typically the seeds of one scenario, when --average_runs is
    off) are drawn in one colour and appear once in the legend; repeats get
    matplotlib's "_nolegend_" sentinel. Otherwise every curve keeps its own
    colour and entry, as before.
    """
    if paper_ready:
        keys = [c["label"] for c in curves]
    else:
        keys = list(range(len(curves)))

    order = []
    for k in keys:
        if k not in order:
            order.append(k)
    n_slots = len(order)

    if n_slots <= 10:
        cmap = plt.cm.get_cmap("tab10")
    elif n_slots <= 20:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = plt.cm.get_cmap("hsv", n_slots + 1)

    colors, legend_labels, seen = [], [], set()
    for curve, k in zip(curves, keys):
        colors.append(cmap(order.index(k)))
        if k in seen:
            legend_labels.append("_nolegend_")
        else:
            seen.add(k)
            legend_labels.append(curve["label"])
    return colors, legend_labels


def _draw_curve(ax, curve, key, color, linestyle, linewidth, alpha=1.0,
                label=None, band_alpha=0.18):
    """Plot one series, with a ±1 std band when the curve is a seed average."""
    if curve.get(key) is None:
        return                              # series absent for this run (e.g. BoN, older data)
    y = np.asarray(curve[key])
    x = np.asarray(curve["x_samples"])[:len(y)]
    ax.plot(x, y, label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)
    std = curve.get(key + "_std")
    if std is not None:
        std = np.asarray(std)[:len(y)]
        ax.fill_between(x, y - std, y + std, color=color, alpha=band_alpha,
                        linewidth=0)


def plot_convergence(filepaths, output_path, num_prompts=None, subtitle=None,
                     paper_ready=False, average_runs=False):
    """Generate Figure 3-style convergence plot.

    Plots reward vs iteration/sample count for multiple methods.

    paper_ready: if True, use minimal legends (just method names; MISVO
    instances disambiguate by regularizer when >1 misvo file is present)
    and save each panel as its own figure alongside the combined view.

    average_runs: if True, files that differ only by their seed token are
    collapsed into one curve — the mean across seeds, with a ±1 std band
    (std taken per x-point across seeds).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve filepaths once so per-panel saving and label lookups see the
    # same expanded list.
    resolved_paths = []
    curves = []
    for fp in filepaths:
        expanded = sorted(glob.glob(fp)) if "*" in fp else [fp]
        for path in expanded:
            if "candidates" in path or "comparison" in path or "all_metrics" in path:
                continue
            curve = _extract_convergence_curves(path, num_prompts=num_prompts)
            if curve is not None:
                curves.append(curve)
                resolved_paths.append(path)
            else:
                logger.warning(f"No convergence data in {path}, skipping")

    if not curves:
        logger.error("No convergence data found in any file")
        return

    if average_runs:
        curves, resolved_paths = _aggregate_curves_by_seed(curves, resolved_paths)

    # Paper-ready labels: minimal method names, with MISVO disambiguated by
    # regularizer when 2+ MISVO files are present in the resolved set.
    if paper_ready:
        _apply_paper_labels(curves, resolved_paths)
    elif average_runs:
        _annotate_seed_counts(curves)

    # Linestyle per method family (visual grouping)
    method_linestyles = {
        "aisp": "-",
        "misvo": "-",
        "bon": "--",
    }

    # Assign unique colors: group curves by method, give each curve within
    # a method a distinct shade so different hyperparams are distinguishable.
    curve_colors, legend_labels = _curve_colors_and_legend(
        curves, plt, paper_ready=paper_ready)

    titles = ["Mean at k", "Best at k", "Best so far"]
    keys = ["mean_at_k", "best_at_k", "best_so_far"]

    # --- Combined 1×3 figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, title, key in zip(axes, titles, keys):
        for i, curve in enumerate(curves):
            _draw_curve(ax, curve, key, curve_colors[i],
                        method_linestyles.get(curve["method"], ":"),
                        linewidth=1.5, alpha=0.85, label=legend_labels[i])
        ax.set_xlabel("Samples")
        ax.set_ylabel("Reward")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.10), fontsize=9)
    if not paper_ready:
        sup = "Sample Efficiency: Reward vs Samples"
        if subtitle:
            sup += f"\n{subtitle}"
        fig.suptitle(sup, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved convergence plot to {output_path}")
    plt.close(fig)

    # --- Paper-ready: one figure per panel ---
    if paper_ready:
        root, ext = os.path.splitext(output_path)
        for title, key in zip(titles, keys):
            fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
            for i, curve in enumerate(curves):
                _draw_curve(ax_p, curve, key, curve_colors[i],
                            method_linestyles.get(curve["method"], ":"),
                            linewidth=2.0, label=legend_labels[i])
            ax_p.set_xlabel("Samples")
            ax_p.set_ylabel("Reward")
            ax_p.set_title(title)
            ax_p.grid(True, alpha=0.3)
            ax_p.legend(loc="best", fontsize=10, frameon=True)
            fig_p.tight_layout()
            panel_path = f"{root}_{key}{ext}"
            fig_p.savefig(panel_path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved per-panel paper plot to {panel_path}")
            plt.close(fig_p)


def plot_binary_convergence(filepaths, output_path, num_prompts=None,
                            subtitle=None, paper_ready=False,
                            average_runs=False):
    """Generate convergence plot using binary reward values.

    Only includes experiments that have binary_mean_per_iter / binary_best_per_iter.
    paper_ready: see plot_convergence — minimal legends + per-panel saving.
    average_runs: see plot_convergence — seed-average curve + ±1 std band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved_paths = []
    curves = []
    for fp in filepaths:
        expanded = sorted(glob.glob(fp)) if "*" in fp else [fp]
        for path in expanded:
            if "candidates" in path or "comparison" in path or "all_metrics" in path:
                continue
            curve = _extract_convergence_curves(path, num_prompts=num_prompts)
            if curve is not None and "binary_mean_at_k" in curve:
                curves.append(curve)
                resolved_paths.append(path)

    if not curves:
        logger.info("No binary convergence data found, skipping binary convergence plot")
        return

    if average_runs:
        curves, resolved_paths = _aggregate_curves_by_seed(curves, resolved_paths)

    if paper_ready:
        _apply_paper_labels(curves, resolved_paths)
    elif average_runs:
        _annotate_seed_counts(curves)

    method_linestyles = {
        "aisp": "-", "misvo": "-", "bon": "--",
    }

    curve_colors, legend_labels = _curve_colors_and_legend(
        curves, plt, paper_ready=paper_ready)

    titles = ["Binary Mean at k", "Binary Best at k", "Binary Best so far",
              "Pass@1 (float-selected)"]
    keys = ["binary_mean_at_k", "binary_best_at_k", "binary_best_so_far",
            "binary_pass1_best_so_far"]

    # --- Combined figure (one panel per key) ---
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    for ax, title, key in zip(axes, titles, keys):
        for i, curve in enumerate(curves):
            _draw_curve(ax, curve, key, curve_colors[i],
                        method_linestyles.get(curve["method"], ":"),
                        linewidth=1.5, alpha=0.85, label=legend_labels[i])
        ax.set_xlabel("Samples")
        ax.set_ylabel("Binary Reward")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.10), fontsize=9)
    if not paper_ready:
        sup = "Sample Efficiency: Binary Reward vs Samples"
        if subtitle:
            sup += f"\n{subtitle}"
        fig.suptitle(sup, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved binary convergence plot to {output_path}")
    plt.close(fig)

    # --- Paper-ready: one figure per panel ---
    if paper_ready:
        root, ext = os.path.splitext(output_path)
        for title, key in zip(titles, keys):
            fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
            for i, curve in enumerate(curves):
                _draw_curve(ax_p, curve, key, curve_colors[i],
                            method_linestyles.get(curve["method"], ":"),
                            linewidth=2.0, label=legend_labels[i])
            ax_p.set_xlabel("Samples")
            ax_p.set_ylabel("Binary Reward")
            ax_p.set_title(title)
            ax_p.grid(True, alpha=0.3)
            ax_p.legend(loc="best", fontsize=10, frameon=True)
            fig_p.tight_layout()
            panel_path = f"{root}_{key}{ext}"
            fig_p.savefig(panel_path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved per-panel paper plot to {panel_path}")
            plt.close(fig_p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--pattern", type=str, default="*.json")
    parser.add_argument("--output", type=str, default=None)
    # Skip candidates files (contain lists of N responses, not final results)
    parser.add_argument("--skip_candidates", action="store_true", default=True)
    # Convergence plot (Figure 3) — generated automatically alongside metrics
    parser.add_argument("--no_plot", action="store_true", default=False,
                        help="Skip convergence plot generation")
    parser.add_argument("--plot_output", type=str, default=None,
                        help="Output path for convergence plot (default: <results_dir>/convergence.png)")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Only evaluate the first N prompts from each results file")
    parser.add_argument("--shared_prompts", action="store_true", default=False,
                        help="Also report metrics using only the prompts shared across all files "
                             "(first min(N) prompts)")
    parser.add_argument("--paper_ready", action="store_true", default=False,
                        help="Use minimal legends (just method names; MISVO instances "
                             "disambiguate by regularizer when >1 misvo file is present) "
                             "and additionally save each convergence panel as its own "
                             "figure (e.g. convergence_mean_at_k.png).")
    parser.add_argument("--average-runs", "--average_runs", dest="average_runs",
                        action="store_true", default=False,
                        help="In the convergence plots, collapse runs that differ "
                             "only by their seed (`_seed42` / `_s42`) into a single "
                             "curve: the mean across seeds plus a shaded ±1 std band "
                             "(std computed per x-point across seeds). Scenarios with "
                             "a single run are drawn as before.")
    parser.add_argument("--no_cache", action="store_true", default=False,
                        help=f"Disable the per-directory result cache "
                             f"(`<results_dir>/{CACHE_FILENAME}`). By default, "
                             f"evaluate_file results are cached by file-content "
                             f"hash + num_prompts; reruns with unchanged inputs "
                             f"reuse the stored metrics instead of recomputing.")
    args = parser.parse_args()

    setup_logging()

    # --- Collect files ---
    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))

    # Filter out non-result files
    skip_patterns = ["candidates", "comparison", "all_metrics", "convergence"]
    files = [f for f in files if not any(s in os.path.basename(f) for s in skip_patterns)]

    if not files:
        logger.error(f"No result files found in {args.results_dir} with pattern {args.pattern}")
        return

    # --- Load result cache (per results_dir; survives across runs) ---
    # The cache key is (sha256(file_content), num_prompts). The regular pass
    # below and the --shared_prompts re-pass both go through the cache and
    # only recompute on hash/N mismatch.
    use_cache = not args.no_cache
    cache_dir = (os.path.dirname(args.file) if args.file
                 else args.results_dir)
    cache = _load_cache(cache_dir) if use_cache else \
            {"version": CACHE_VERSION, "entries": {}}

    # --- Metrics evaluation ---
    all_metrics = []
    for path in files:
        logger.info(f"Evaluating {path}...")
        try:
            metrics = _evaluate_file_cached(
                path, args.num_prompts, cache, use_cache=use_cache)
            all_metrics.append(metrics)
            logger.info(
                f"  Reward: {metrics['reward_mean']:.4f} ({metrics['reward_std']:.4f}), "
                f"Diversity: {metrics['diversity_mean']:.4f}, "
                f"Coherence: {metrics.get('coherence_mean', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {path}: {e}")

    # Flush cache after the regular pass (in case the shared-prompts pass
    # fails partway and we want at least the per-file entries persisted).
    if use_cache:
        _save_cache(cache_dir, cache)

    # --- Shared prompts evaluation ---
    shared_n = None
    if args.shared_prompts and len(all_metrics) > 1:
        shared_n = min(m["num_samples"] for m in all_metrics)
        if args.num_prompts is not None:
            shared_n = min(shared_n, args.num_prompts)
        logger.info(f"Shared prompts: re-evaluating all files with first "
                     f"{shared_n} prompts")
        for path, m in zip(files, all_metrics):
            try:
                sm = _evaluate_file_cached(
                    path, shared_n, cache, use_cache=use_cache)
                m["shared_n"] = shared_n
                m["shared_reward_mean"] = sm["reward_mean"]
                m["shared_float_reward_mean"] = sm.get("float_reward_mean")
            except Exception as e:
                logger.error(f"Shared-prompts eval failed for {path}: {e}")
                m["shared_n"] = shared_n
                m["shared_reward_mean"] = None
                m["shared_float_reward_mean"] = None
        if use_cache:
            _save_cache(cache_dir, cache)

    # Print summary table
    has_float_rewards = any(m.get("float_reward_mean") is not None for m in all_metrics)
    has_shared = shared_n is not None
    name_w = max(len(m["file"]) for m in all_metrics) + 2
    name_w = max(name_w, 6)  # at least "File" + padding

    # Build columns: (header, width, format_fn)
    columns = []
    columns.append(("N", 5, lambda m: f"{m['num_samples']}"))
    if has_float_rewards:
        columns.append(("Binary Rw", 12, lambda m: f"{m['reward_mean']:.4f}" if m['reward_mean'] is not None else "N/A"))
        columns.append(("Float Rw", 12, lambda m: f"{m['float_reward_mean']:.4f}" if m.get('float_reward_mean') is not None else "N/A"))
        if has_shared:
            sh_label = f"Shared({shared_n})"
            columns.append((sh_label, max(12, len(sh_label) + 2),
                            lambda m: f"{m['shared_float_reward_mean']:.4f}" if m.get('shared_float_reward_mean') is not None else "N/A"))
    else:
        columns.append(("Reward", 12, lambda m: f"{m['reward_mean']:.4f}" if m['reward_mean'] is not None else "N/A"))
        if has_shared:
            sh_label = f"Shared({shared_n})"
            columns.append((sh_label, max(12, len(sh_label) + 2),
                            lambda m: f"{m['shared_reward_mean']:.4f}" if m.get('shared_reward_mean') is not None else "N/A"))
    columns.append(("Diversity", 12, lambda m: f"{m['diversity_mean']:.4f}"))
    columns.append(("Coherence", 12, lambda m: f"{m.get('coherence_mean', None):.4f}" if m.get('coherence_mean') is not None else "N/A"))

    w = name_w + sum(c[1] for c in columns)
    print("\n" + "=" * w)
    header = f"{'File':<{name_w}}" + "".join(f"{c[0]:>{c[1]}}" for c in columns)
    print(header)
    print("-" * w)
    for m in all_metrics:
        row = f"{m['file']:<{name_w}}" + "".join(f"{c[2](m):>{c[1]}}" for c in columns)
        print(row)
    print("=" * w)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {args.output}")

    # --- CSV output ---
    if args.file:
        csv_dir = os.path.dirname(args.file) or "results"
    else:
        csv_dir = args.results_dir
    csv_path = os.path.join(csv_dir, "results.csv")
    import csv
    csv_columns = ["file", "num_samples", "reward_mean", "reward_std",
                   "float_reward_mean", "float_reward_std",
                   "shared_n", "shared_reward_mean", "shared_float_reward_mean",
                   "diversity_mean", "diversity_std",
                   "rep_2", "rep_3", "rep_4",
                   "coherence_mean", "coherence_std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"Saved CSV to {csv_path}")

    # --- Convergence plot (Figure 3) ---
    if not args.no_plot:
        plot_output = args.plot_output
        if plot_output is None:
            if args.file:
                plot_dir = os.path.dirname(args.file) or "results"
            else:
                plot_dir = args.results_dir
            plot_output = os.path.join(plot_dir, "convergence.png")

        conv_num_prompts = args.num_prompts
        conv_subtitle = None
        if args.shared_prompts and shared_n is not None:
            conv_num_prompts = shared_n
            conv_subtitle = f"(first {shared_n} shared prompts)"

        plot_convergence(files, plot_output, num_prompts=conv_num_prompts,
                         subtitle=conv_subtitle, paper_ready=args.paper_ready,
                         average_runs=args.average_runs)

        # Binary convergence plot (only includes experiments with binary per-iter data)
        binary_plot_output = plot_output.replace(".png", "_binary.png")
        plot_binary_convergence(files, binary_plot_output,
                                num_prompts=conv_num_prompts,
                                subtitle=conv_subtitle,
                                paper_ready=args.paper_ready,
                                average_runs=args.average_runs)



if __name__ == "__main__":
    main()
