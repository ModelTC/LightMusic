#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
source "${PROJECT_ROOT}/scripts/environment.sh"
cd "$PROJECT_ROOT"
exec "${MUSIC_COVER_BASE_PREFIX}/bin/python" -m music_cover.rvc_training --stage 00_prepare "$@"
