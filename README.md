# MISVO: Minimally Invasive Steering of Language Models

Code for [*Minimally Invasive Steering of Language Models*](https://arxiv.org/abs/2609.30218)
(Taha Entesari, Jingyu Zhang, Daniel Khashabi, Mahyar Fazlyab).

MISVO adapts a frozen language model to a test-time reward by optimizing
position-specific steering vectors `u_t` added to the hidden state that enters
the LM head. The reward gradient is a REINFORCE estimate whose advantages are
the rewards z-scored within each step's `K` rollouts; the regularizer is the Fisher quadratic `½ Σ_t u_tᵀ F̄_t u_t` of the
reference policy, estimated from the first step's rollouts (at `u = 0`) and
applied through matrix-free Fisher–vector products with the frozen LM head.
No model parameters are updated.

The repository contains MISVO, the two baselines from the paper (Best-of-N with
top-p sampling and AISP), the SHP and MBPP+ tasks, and the evaluation and
table/figure scripts.

## Installation

```bash
conda create -n misvo python=3.10 -y && conda activate misvo
pip install -r requirements.txt
pip install --no-deps simcse==0.4        # coherence metric only
python data/mbpp_plus/download.py        # MBPP+ problems -> data/mbpp_plus/mbppplus.jsonl
```

Models and the SHP dataset are downloaded from the Hugging Face Hub on first
use (`google/gemma-3-4b-it` and `meta-llama/Meta-Llama-3-8B-Instruct` are gated:
accept their licenses and `huggingface-cli login` first).

## Running

One method, one config, one seed:

```bash
python scripts/run_misvo.py --config configs/lfm25_1b_shp.yaml --num_prompts 500 --seed 42 \
    --rm_device cuda --incremental_save --output_dir results/lfm25_1b_shp
python scripts/run_aisp.py  --config configs/lfm25_1b_shp.yaml --num_prompts 500 --seed 42 \
    --rm_device cuda --incremental_save --output_dir results/lfm25_1b_shp --compute_regularizers
python scripts/run_bon.py   --config configs/lfm25_1b_shp.yaml --num_prompts 500 --seed 42 \
    --rm_device cuda --incremental_save --output_dir results/lfm25_1b_shp --variant topp
```

or all three with the paper's settings:

```bash
CONFIG=configs/lfm25_1b_shp.yaml SEED=42 bash scripts/run_experiment.sh
```

Configs: `configs/{lfm25_1b,gemma3_4b_it,llama3_8b_instruct,phi4}_shp.yaml` and
`configs/{lfm25_1b,gemma3_4b_it,llama3_8b_instruct}_mbppplus.yaml`. Every
config value can be overridden on the command line (`--lr`, `--lam`,
`--n_samples`, `--n_steps`, `--regularizer`, ...; see `--help`).

Each run writes `<method>_<model>_<dataset>_<hyperparameters>_seed<S>.json`
(one entry per prompt: best response and reward, per-step mean/max reward,
steering-vector norm, KL-to-reference and Fisher diagnostics) and a matching
`*_generations.json` with every sampled response and its reward.

### Throughput options

All are on the code paths used for the paper and do not change the algorithm.

| Option | What it does | Where |
|---|---|---|
| `--wave_size W` | batches the `K` rollouts of `W` prompts into one `generate()` call; each prompt keeps its own `u` and Adam state | `run_misvo_wave` in `src/methods/misvo.py`, `*_wave` in `src/models/base_llm.py`, `aisp.py`, `best_of_n.py` |
| `--num_gpus G` | one worker per GPU, each with its own model and reward-model copy, pulling prompts from a shared queue | `src/multi_gpu.py` |
| `--n_chunks`, `--gen_chunks` | position chunking of the gradient computation and rollout chunking of sampling; both grow automatically on CUDA OOM | `src/methods/misvo.py` |
| (always on) | prompt KV cache reused across all sampling calls; perturbations injected by a forward hook so HF `generate()` does the decoding; matrix-free Fisher–vector products (no `d×d` matrices); split einsums that avoid `(K, T, V)` intermediates | `src/models/base_llm.py`, `src/methods/misvo.py` |
| `--incremental_save` | atomic per-prompt saves; re-running the same command resumes | `src/utils.py` |
| `--shard_id/--num_shards`, `--start_idx/--end_idx` | split the prompt set across jobs (auto-detected inside 0-indexed Slurm arrays); merge with `scripts/merge_shards.py` | `src/utils.py` |

## Reproducing the paper

`scripts/reproduce_paper.sh` lists every run behind the tables (written
sequentially; on a cluster submit each line as a job), followed by:

```bash
python scripts/make_tables.py --results_dir results          # Tables 1, 2, 3, 5
python scripts/evaluate.py --results_dir results/lfm25_1b_shp --paper_ready --average-runs   # trajectory figures
python scripts/profile_method.py --config configs/lfm25_1b_shp.yaml --method misvo \
    --num_prompts 10 --warmup 5                               # Table 4 (one GPU)
```

`make_tables.py` computes rewards from the result files and diversity /
coherence with `scripts/evaluate.py` (cached per directory). For MBPP+ it
re-scores the selected programs once against the full MBPP+ test suite and
writes the plus-suite scores back into the result files.

### Settings used for the reported results

| | LFM2.5-1.2B | Gemma3-4B-IT | Llama-3-8B-Instruct | Phi-4 |
|---|---|---|---|---|
| rollouts per step `K` = steps `N` | 16 | 16 | 32 | 32 |
| BoN budget | 256 | 256 | 1024 | 1024 |
| MISVO `λ` | 1.0 | 1.0 | 3.0 | 3.0 |
| weights dtype | fp16 | bf16 | fp16 | fp16 |

Common to all: MISVO learning rate 0.1 (Adam), temperature 0.7, top-p 0.95,
per-step reward z-scoring (`reward_norm: batch`), 512 new tokens; AISP `σ² = 0.5`, `α = 0.9999`,
`λ = 0.1` (MBPP+ also `λ = 1`); seeds 42, 1340, 2026. SHP uses 500 test prompts
drawn with the run's seed (so each seed uses a different prompt sample);
MBPP+ uses the first 120 problems.

After the `N` optimization steps MISVO and AISP each draw one more response
from the final policy (greedy with the optimized `u` for MISVO); it is
included in the best-so-far selection.

### Regularizer names

| `--regularizer` | Paper |
|---|---|
| `frozen_fisher` (default) | MISVO, frozen-reference Fisher `F̄_t` |
| `steered_traj_fisher` | on-policy rollouts, off-policy Fisher `F̄_t^u` |
| `fully_adaptive_fisher` | on-policy rollouts, on-policy Fisher `F̃_t^u` |
| `kl` | direct KL regularizer (sampled log-ratio folded into the reward) |
| `l2` | hidden-space `ℓ2` |
| any, with `--lam 0` | unregularized steering |

## Code map

```
src/methods/misvo.py       MISVO (single prompt and wave-batched)
src/methods/aisp.py        AISP baseline, reward normalization, KL-to-reference diagnostic
src/methods/best_of_n.py   Best-of-N (top-p) baseline
src/models/base_llm.py     frozen LM wrapper: pre-logit perturbation hooks, KV-cache reuse, wave generation
src/models/reward_model.py Skywork-Reward-V2 scorer
src/evaluation/            MBPP+ unit-test scorer, diversity, coherence
src/data/datasets.py       SHP and MBPP+ prompts and prompt formatting
src/multi_gpu.py           multi-GPU prompt-parallel workers
scripts/run_{misvo,aisp,bon}.py   experiment runners
scripts/evaluate.py        metrics and trajectory plots
scripts/make_tables.py     paper tables
scripts/profile_method.py  runtime / memory profiling
```

## Citation

```bibtex
@article{entesari2026misvo,
  title   = {Minimally Invasive Steering of Language Models},
  author  = {Entesari, Taha and Zhang, Jingyu and Khashabi, Daniel and Fazlyab, Mahyar},
  journal = {arXiv preprint arXiv:2609.30218},
  year    = {2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
