#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/environment.sh"
exec "${PROJECT_ROOT}/envs/control/bin/python" -m music_cover.pipeline "$@"
