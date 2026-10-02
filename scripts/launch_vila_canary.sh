#!/bin/bash
set -euo pipefail

PATHOTME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGVL_REPO_ROOT="${PGVL_REPO_ROOT:-/path/to/PGVL-Gym}"
PATHOTME_RESULTS_ROOT="${PATHOTME_RESULTS_ROOT:-/path/to/shared/PathoTME-results}"
PGVL_ACCOUNT="${PGVL_ACCOUNT:-shared-delta-gpu}"
LOG_DIR="${PATHOTME_RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi
if [[ $# -gt 0 ]]; then
    folds=("$@")
else
    folds=(0 2 3)
fi
for fold in "${folds[@]}"; do
    if [[ ! "${fold}" =~ ^[0-4]$ ]]; then
        echo "usage: $0 [--dry-run] [fold ...]" >&2
        exit 2
    fi
done

for fold in "${folds[@]}"; do
    command=(
        sbatch
        --parsable
        --account="${PGVL_ACCOUNT}"
        --job-name="pathotme-vila-f${fold}"
        --chdir="${PATHOTME_ROOT}"
        --output="${LOG_DIR}/pathotme-vila-f${fold}-%j.out"
        --export="ALL,PATHOTME_ROOT=${PATHOTME_ROOT},PATHOTME_FOLD=${fold},PGVL_REPO_ROOT=${PGVL_REPO_ROOT},PATHOTME_RESULTS_ROOT=${PATHOTME_RESULTS_ROOT}"
        "${PATHOTME_ROOT}/scripts/slurm/vila_fusion.slurm"
    )
    if [[ "${dry_run}" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        job_id="$("${command[@]}")"
        printf 'fold=%s job_id=%s\n' "${fold}" "${job_id}"
    fi
done

if [[ "${dry_run}" == true ]]; then
    exit 0
fi
if [[ ${#folds[@]} -eq 0 ]]; then
    echo "no folds selected" >&2
    exit 2
fi
