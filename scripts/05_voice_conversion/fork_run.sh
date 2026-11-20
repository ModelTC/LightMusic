#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/environment.sh"
exec "${MUSIC_COVER_BASE_PREFIX}/bin/python" "${PROJECT_ROOT}/scripts/fork_voice_run.py" "$@"
