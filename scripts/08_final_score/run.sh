#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/environment.sh"
exec bash "${PROJECT_ROOT}/scripts/run_baseline.sh" --stage 08_final_score "$@"
