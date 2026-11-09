"""
Rebuild the paper's result tables from result JSONs.

  Table 1  SHP: BoN / AISP / MISVO x 4 models — reward, diversity, coherence
           (mean ± std over seeds).
  Table 2  MBPP+: BoN / AISP (λ=0.1, 1) / MISVO x 3 models — pass rate (fraction
           of problems whose selected program passes the full MBPP+ test
           suite), diversity, coherence (mean ± std over seeds).
  Table 3  Policy deviation on LFM2.5/SHP (100 prompts, seed 42) — reward,
           final ||u||, final KL-to-reference (± across prompts).
  Table 5  Regularizer ablation on LFM2.5/SHP (seed 42) — reward, diversity,
           coherence.

Result files are located by their output name anywhere under --results_dir
(run_*.py write e.g. results/<config>/misvo_lfm25_1b_shp_..._seed42.json), so
the directory layout does not matter. SHP reward is the mean over prompts of
the stored best-so-far `reward` (best of all K*N rollouts and the final
response). Diversity, coherence and the MBPP+ pass rates come from
scripts/evaluate.py (with its per-directory cache); evaluate.py re-scores MBPP+
files once against the full plus test suite and writes the scores back.

Usage:
    python scripts/make_tables.py --results_dir results
    python scripts/make_tables.py --results_dir results --tables 1 3 --no_coherence
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import evaluate as ev  # noqa: E402  (scripts/evaluate.py)
from src.utils import setup_logging, logger  # noqa: E402

SEEDS = (42, 1340, 2026)
# model -> (display name, K = rollouts per step = steps, MISVO lambda)
MODELS = {
    "lfm25_1b": ("LFM2.5 1.2B", 16, 1.0),
    "gemma3_4b_it": ("Gemma3 4B", 16, 1.0),
    "llama3_8b_instruct": ("Llama3 8B Instruct", 32, 3.0),
    "phi4": ("Phi 4 (14B)", 32, 3.0),
}
SKIP = ("_generations", "_candidates", "_vectors", "_errors", "_shard", "_range", "_fix")


def _patterns(model, dataset, seed, lam_aisp=0.1, reg="frozen_fisher", lam_misvo=None):
    """Filename regexes for the three methods."""
    _, K, lam_default = MODELS[model]
    lam_misvo = lam_default if lam_misvo is None else lam_misvo
    frw = r"_floatrw" if dataset == "mbpp_plus" else ""
    return {
        "bon": rf"^bon_topp_{model}_{dataset}_N{K * K}_t0\.7_p0\.95{frw}_seed{seed}\.json$",
        "aisp": rf"^aisp_{model}_{dataset}_n{K}_k{K}_lam{lam_aisp}_a0\.9999_sig0\.5_normbatch{frw}_seed{seed}\.json$",
        "misvo": rf"^misvo_{model}_{dataset}_ns{K}_steps{K}_lr0\.1_lam{lam_misvo}_reg{reg}_normbatch{frw}_seed{seed}\.json$",
    }


class Finder:
    def __init__(self, root, exclude_dir=None):
        files = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
        excl = os.path.abspath(exclude_dir) if exclude_dir else None
        self.files = [f for f in files
                      if not any(s in os.path.basename(f) for s in SKIP)
                      and not os.path.basename(f).startswith(".")
                      and (excl is None or not os.path.abspath(f).startswith(excl + os.sep))]

    def find(self, pattern, min_entries=1):
        """The matching file with the most entries (None if nothing matches)."""
        hits = [f for f in self.files if re.match(pattern, os.path.basename(f))]
        best, best_n = None, 0
        for f in hits:
            n = _num_entries(f)
            if n >= min_entries and n > best_n:
                best, best_n = f, n
        if len(hits) > 1:
            logger.info(f"  {len(hits)} files match {pattern}; using {best} ({best_n} entries)")
        return best


def _num_entries(path):
    try:
        with open(path) as f:
            return len(json.load(f))
    except (OSError, ValueError, TypeError):
        return 0


class Metrics:
    """evaluate.py metrics with its per-directory cache."""

    def __init__(self, coherence=True):
        self.caches = {}
        if not coherence:
            ev.compute_coherence_batch = lambda prompts, responses: {
                "coherence_mean": None, "coherence_std": None}

    def __call__(self, path, num_prompts=None):
        d = os.path.dirname(path)
        if d not in self.caches:
            self.caches[d] = ev._load_cache(d)
        m = ev._evaluate_file_cached(path, num_prompts, self.caches[d], use_cache=True)
        ev._save_cache(d, self.caches[d])
        return m


def _fmt(vals, digits=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f} (1 seed)"
    tag = "" if len(vals) == len(SEEDS) else f" ({len(vals)} seeds)"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}{tag}"


def _print_table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def table_seeded(finder, metrics, dataset, models, methods, reward_key, digits):
    rows = []
    for model in models:
        for label, method, kw in methods:
            rew, div, coh = [], [], []
            for seed in SEEDS:
                path = finder.find(_patterns(model, dataset, seed, **kw)[method])
                if path is None:
                    continue
                m = metrics(path)
                if reward_key == "stored_reward":
                    with open(path) as f:
                        rew.append(float(np.mean([e["reward"] for e in json.load(f)])))
                else:
                    rew.append(m[reward_key])
                div.append(m["diversity_mean"])
                coh.append(m.get("coherence_mean"))
            rows.append([MODELS[model][0], label, _fmt(rew, digits), _fmt(div, 3), _fmt(coh, 3)])
    return rows


def table1(finder, metrics):
    methods = [("BoN (top-p)", "bon", {}), ("AISP", "aisp", {}), ("MISVO", "misvo", {})]
    rows = table_seeded(finder, metrics, "shp", list(MODELS), methods, "stored_reward", 2)
    _print_table("Table 1 — SHP (Skywork-Reward-V2-Qwen3-0.6B, best-so-far reward)",
                 ["Model", "Method", "Reward", "Diversity", "Coherence"], rows)


def table2(finder, metrics):
    methods = [("BoN (top-p)", "bon", {}),
               ("AISP (λ=0.1)", "aisp", {"lam_aisp": 0.1}),
               ("AISP (λ=1)", "aisp", {"lam_aisp": 1.0}),
               ("MISVO", "misvo", {})]
    models = ["lfm25_1b", "gemma3_4b_it", "llama3_8b_instruct"]
    for key, name in (("reward_mean", "pass rate: selected program passes all MBPP+ tests"),
                      ("float_reward_mean", "supplementary: mean fraction of MBPP+ tests passed")):
        rows = table_seeded(finder, metrics, "mbpp_plus", models, methods, key, 3)
        _print_table(f"Table 2 — MBPP+ ({name})",
                     ["Model", "Method", "Pass rate", "Diversity", "Coherence"], rows)


def table3(finder3, metrics):
    rows = []
    for label, method, kw in [("MISVO", "misvo", {}), ("AISP", "aisp", {}),
                              ("Unregularized steering", "misvo", {"lam_misvo": 0.0}),
                              ("Direct KL regularizer", "misvo", {"reg": "kl"}),
                              ("Hidden-space ℓ2", "misvo", {"reg": "l2"})]:
        path = finder3.find(_patterns("lfm25_1b", "shp", 42, **kw)[method])
        if path is None:
            rows.append([label, "n/a", "n/a", "n/a"])
            continue
        with open(path) as f:
            data = json.load(f)
        r = np.array([e["reward"] for e in data], dtype=float)
        if method == "aisp":  # AISP logs l2 = ||u||^2 and token_kl_per_iter
            u = np.sqrt([e["l2_per_iter"][-1] for e in data])
            kl = np.array([e["token_kl_per_iter"][-1] for e in data], dtype=float)
        else:
            u = np.array([e["u_norm_per_iter"][-1] for e in data], dtype=float)
            kl = np.array([e["kl_per_iter"][-1] for e in data], dtype=float)
        rows.append([label, f"{r.mean():.2f} ± {r.std():.2f}", f"{u.mean():.1f} ± {u.std():.1f}",
                     f"{kl.mean():.1f} ± {kl.std():.1f}"])
    # BoN with the same budget (K*N = 256) on the same 100 prompts.
    path = finder3.find(_patterns("lfm25_1b", "shp", 42)["bon"]) or \
        Finder(finder3.root_all).find(_patterns("lfm25_1b", "shp", 42)["bon"])
    if path is not None:
        n = min(_num_entries(path), 100)
        m = metrics(path, num_prompts=n)
        rows.append([f"BoN (N=256, first {n} prompts)",
                     f"{m['reward_mean']:.2f} ± {m['reward_std']:.2f}", "n/a", "n/a"])
    _print_table("Table 3 — policy deviation, LFM2.5-1.2B / SHP (± across prompts)",
                 ["Method / regularizer", "Reward", "Final ||u||", "Final KL"], rows)


def table5(finder, metrics):
    rows = []
    for label, reg in [("off-policy rollouts, off-policy Fisher F̄_t (MISVO)", "frozen_fisher"),
                       ("on-policy rollouts, off-policy Fisher F̄_t^u", "steered_traj_fisher"),
                       ("on-policy rollouts, on-policy Fisher F̃_t^u", "fully_adaptive_fisher"),
                       ("KL", "kl")]:
        path = finder.find(_patterns("lfm25_1b", "shp", 42, reg=reg)["misvo"])
        if path is None:
            rows.append([label, "n/a", "n/a", "n/a"])
            continue
        m = metrics(path)
        coh = m.get("coherence_mean")
        with open(path) as f:
            reward = float(np.mean([e["reward"] for e in json.load(f)]))
        rows.append([label, f"{reward:.2f}", f"{m['diversity_mean']:.3f}",
                     "n/a" if coh is None else f"{coh:.3f}"])
    _print_table("Table 5 — regularizer ablation, LFM2.5-1.2B / SHP, seed 42",
                 ["Regularizer", "Reward", "Diversity", "Coherence"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--deviation_dir", type=str, default=None,
                        help="Directory with the 100-prompt Table 3 runs "
                             "(default: <results_dir>/lfm25_1b_shp_deviation). It is "
                             "excluded when locating Table 1/5 runs.")
    parser.add_argument("--tables", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--no_coherence", action="store_true",
                        help="Skip SimCSE coherence (no GPU / simcse package needed)")
    args = parser.parse_args()
    setup_logging()

    dev_dir = args.deviation_dir or os.path.join(args.results_dir, "lfm25_1b_shp_deviation")
    finder = Finder(args.results_dir, exclude_dir=dev_dir)
    finder3 = Finder(dev_dir) if os.path.isdir(dev_dir) else Finder(args.results_dir)
    finder3.root_all = args.results_dir
    metrics = Metrics(coherence=not args.no_coherence)

    if 1 in args.tables:
        table1(finder, metrics)
    if 2 in args.tables:
        table2(finder, metrics)
    if 3 in args.tables:
        table3(finder3, metrics)
    if 5 in args.tables:
        table5(finder, metrics)


if __name__ == "__main__":
    main()
