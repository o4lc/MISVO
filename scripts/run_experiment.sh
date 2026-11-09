#!/bin/bash
# Run any subset of {bon, aisp, misvo} for one config and one seed with the
# paper's settings. Results go to results/<config name>/.
#
#   CONFIG=configs/lfm25_1b_shp.yaml SEED=42 bash scripts/run_experiment.sh
#   CONFIG=configs/lfm25_1b_mbppplus.yaml METHODS="misvo" SEED=1340 bash scripts/run_experiment.sh
#   CONFIG=configs/phi4_shp.yaml NUM_GPUS=4 bash scripts/run_experiment.sh
#
# Environment variables (all optional except CONFIG):
#   METHODS      methods to run, in order               (default: "bon aisp misvo")
#   SEED         seed; on SHP it also selects the prompts (default: 42)
#   NUM_PROMPTS  default 500 for SHP, 120 for MBPP+
#   OUTPUT_DIR   default results/<config name>
#   NUM_GPUS     data-parallel workers, one prompt stream per GPU (default: 1)
#   WAVE_SIZE    prompts batched together per generate() call (default: 1)
#   AISP_LAMS    AISP lambdas to run (default: "0.1" on SHP, "0.1 1.0" on MBPP+)
#   EXTRA_ARGS   passed verbatim to every runner (e.g. "--start_idx 0 --end_idx 250")
#
# The runners resume from incremental saves, so re-running the same command
# continues an interrupted run. Inside a Slurm *array* job the runners split the
# prompts across array tasks automatically (0-indexed arrays: --array=0-<N-1>);
# merge the shards afterwards with scripts/merge_shards.py.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CONFIG:?set CONFIG=configs/<model>_<task>.yaml}"
NAME=$(basename "$CONFIG" .yaml)
METHODS="${METHODS:-bon aisp misvo}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${NAME}}"
NUM_GPUS="${NUM_GPUS:-1}"
WAVE_SIZE="${WAVE_SIZE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "$NAME" == *_mbppplus ]]; then
    NUM_PROMPTS="${NUM_PROMPTS:-120}"
    AISP_LAMS="${AISP_LAMS:-0.1 1.0}"
    RM_ARGS=()
else
    NUM_PROMPTS="${NUM_PROMPTS:-500}"
    AISP_LAMS="${AISP_LAMS:-0.1}"
    RM_ARGS=(--rm_device cuda)
fi

COMMON=(--config "$CONFIG" --num_prompts "$NUM_PROMPTS" --seed "$SEED"
        --output_dir "$OUTPUT_DIR" --incremental_save
        --num_gpus "$NUM_GPUS" --wave_size "$WAVE_SIZE" "${RM_ARGS[@]}")
mkdir -p "$OUTPUT_DIR"

for method in $METHODS; do
    echo "=== ${method}  config=${CONFIG}  seed=${SEED}  prompts=${NUM_PROMPTS}"
    case "$method" in
        bon)   python scripts/run_bon.py --variant topp "${COMMON[@]}" $EXTRA_ARGS ;;
        aisp)  for lam in $AISP_LAMS; do
                   python scripts/run_aisp.py "${COMMON[@]}" --lam "$lam" --compute_regularizers $EXTRA_ARGS
               done ;;
        misvo) python scripts/run_misvo.py "${COMMON[@]}" $EXTRA_ARGS ;;
        *)     echo "unknown method: $method" >&2; exit 1 ;;
    esac
done
