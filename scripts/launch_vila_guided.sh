#!/bin/bash
set -euo pipefail

PATHOTME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGVL_REPO_ROOT="${PGVL_REPO_ROOT:-/path/to/PGVL-Gym}"
PATHOTME_RESULTS_ROOT="${PATHOTME_RESULTS_ROOT:-/path/to/shared/PathoTME-results}"
PGVL_ACCOUNT="${PGVL_ACCOUNT:-shared-delta-gpu}"
LOG_DIR="${PATHOTME_RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

dry_run=false
mode=both
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=true
            shift
            ;;
        --mode)
            mode="${2:?--mode requires actual, zero, or both}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown option: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done
if [[ "${mode}" == both ]]; then
    modes=(actual zero)
elif [[ "${mode}" == actual || "${mode}" == zero ]]; then
    modes=("${mode}")
else
    echo "--mode must be actual, zero, or both" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    folds=("$@")
else
    folds=(0 2 3)
fi
for fold in "${folds[@]}"; do
    if [[ ! "${fold}" =~ ^(0|2|3)$ ]]; then
        echo "only exact-coverage folds 0, 2, and 3 are allowed" >&2
        exit 2
    fi
done

for tme_mode in "${modes[@]}"; do
    short_mode="${tme_mode:0:1}"
    for fold in "${folds[@]}"; do
        command=(
            sbatch
            --parsable
            --account="${PGVL_ACCOUNT}"
            --job-name="ptme-g-${short_mode}-f${fold}"
            --chdir="${PATHOTME_ROOT}"
            --output="${LOG_DIR}/pathotme-guided-${tme_mode}-f${fold}-%j.out"
            --export="ALL,PATHOTME_ROOT=${PATHOTME_ROOT},PATHOTME_FOLD=${fold},PATHOTME_TME_MODE=${tme_mode},PGVL_REPO_ROOT=${PGVL_REPO_ROOT},PATHOTME_RESULTS_ROOT=${PATHOTME_RESULTS_ROOT}"
            "${PATHOTME_ROOT}/scripts/slurm/vila_guided.slurm"
        )
        if [[ "${dry_run}" == true ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            job_id="$("${command[@]}")"
            printf 'mode=%s fold=%s job_id=%s\n' \
                "${tme_mode}" "${fold}" "${job_id}"
        fi
    done
done
