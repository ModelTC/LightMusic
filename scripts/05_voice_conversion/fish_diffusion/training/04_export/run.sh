#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "${PROJECT_ROOT}/scripts/environment.sh"
export PYTHONDONTWRITEBYTECODE=1
exec "${MUSIC_COVER_BASE_PREFIX}/bin/python" -m music_cover.fish_training --stage 04_export "$@"
