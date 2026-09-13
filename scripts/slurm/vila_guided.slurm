#!/bin/bash
#SBATCH --job-name=pathotme-guided
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --requeue

set -euo pipefail

PATHOTME_ROOT="${PATHOTME_ROOT:?PATHOTME_ROOT must be exported}"
PATHOTME_FOLD="${PATHOTME_FOLD:?PATHOTME_FOLD must be exported}"
PATHOTME_TME_MODE="${PATHOTME_TME_MODE:?PATHOTME_TME_MODE must be exported}"
PGVL_REPO_ROOT="${PGVL_REPO_ROOT:-/path/to/PGVL-Gym}"
if [[ -f "${PGVL_REPO_ROOT}/.env" ]]; then
    set -a
    source "${PGVL_REPO_ROOT}/.env"
    set +a
fi

PATHOTME_DATA_ROOT="${PATHOTME_DATA_ROOT:-/path/to/shared/PathoTME-data}"
PATHOTME_RESULTS_ROOT="${PATHOTME_RESULTS_ROOT:-/path/to/shared/PathoTME-results}"
PATHOTME_PYTHON="${PGVL_CONDA_ENV:?PGVL_CONDA_ENV is unset}/bin/python"
if [[ ! -x "${PATHOTME_PYTHON}" ]]; then
    echo "FATAL: Python is not executable: ${PATHOTME_PYTHON}" >&2
    exit 78
fi

export PATHOTME_DATA_ROOT PATHOTME_RESULTS_ROOT PGVL_REPO_ROOT
export PYTHONPATH="${PATHOTME_ROOT}:${PGVL_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PGVL_STORAGE_ROOT}/.cache_huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PATHOTME_ROOT}"
mkdir -p "${PATHOTME_RESULTS_ROOT}/logs"

guided_args=(
    --experiment-config configs/vila_mil_tme_guided_nsclc_16shot.yaml
    --fold "${PATHOTME_FOLD}"
    --tme-mode "${PATHOTME_TME_MODE}"
    --device cuda:0
)
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    # A preempted attempt can leave an intentionally incomplete result
    # directory. Restart that same fold from a clean adapter initialization.
    guided_args+=(--rerun)
fi

"${PATHOTME_PYTHON}" scripts/run_vila_guided.py "${guided_args[@]}"
