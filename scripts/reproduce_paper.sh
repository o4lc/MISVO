#!/bin/bash
# Every run behind the paper's tables, followed by the table/figure scripts.
# Runs sequentially as written; on a cluster, submit each line as its own job
# (Llama-3-8B and Phi-4 used 4 GPUs: add NUM_GPUS=4).
set -euo pipefail
cd "$(dirname "$0")/.."

# Tables 1 and 2 (+ trajectory figures): 3 seeds x {BoN, AISP, MISVO}.
for seed in 42 1340 2026; do
    for cfg in lfm25_1b_shp gemma3_4b_it_shp llama3_8b_instruct_shp phi4_shp \
               lfm25_1b_mbppplus gemma3_4b_it_mbppplus llama3_8b_instruct_mbppplus; do
        CONFIG=configs/${cfg}.yaml SEED=$seed bash scripts/run_experiment.sh
    done
done

# Table 5: Fisher-surrogate / KL ablation on LFM2.5 SHP, seed 42 (the MISVO row
# is the seed-42 run above).
for reg in steered_traj_fisher fully_adaptive_fisher kl; do
    CONFIG=configs/lfm25_1b_shp.yaml METHODS=misvo EXTRA_ARGS="--regularizer $reg" \
        bash scripts/run_experiment.sh
done

# Table 3: policy deviation on 100 LFM2.5 SHP prompts, seed 42.
DEV=results/lfm25_1b_shp_deviation
CONFIG=configs/lfm25_1b_shp.yaml NUM_PROMPTS=100 OUTPUT_DIR=$DEV METHODS=aisp bash scripts/run_experiment.sh
CONFIG=configs/lfm25_1b_shp.yaml NUM_PROMPTS=100 OUTPUT_DIR=$DEV METHODS=misvo bash scripts/run_experiment.sh
CONFIG=configs/lfm25_1b_shp.yaml NUM_PROMPTS=100 OUTPUT_DIR=$DEV METHODS=misvo EXTRA_ARGS="--lam 0.0" bash scripts/run_experiment.sh
for reg in kl l2; do
    CONFIG=configs/lfm25_1b_shp.yaml NUM_PROMPTS=100 OUTPUT_DIR=$DEV METHODS=misvo \
        EXTRA_ARGS="--regularizer $reg" bash scripts/run_experiment.sh
done

# Table 4: runtime / memory (10 timed prompts after 5 warmup, one GPU).
for cfg in lfm25_1b_shp gemma3_4b_it_shp; do
    for method in bon aisp misvo; do
        python scripts/profile_method.py --config configs/${cfg}.yaml --method $method \
            --num_prompts 10 --warmup 5 --no_torch_profiler
    done
done

# Tables and figures.
python scripts/make_tables.py --results_dir results
for d in results/*_shp; do
    python scripts/evaluate.py --results_dir "$d" --paper_ready --average-runs
done
