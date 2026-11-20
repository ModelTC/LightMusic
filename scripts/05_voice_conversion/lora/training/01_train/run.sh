#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
source "${PROJECT_ROOT}/scripts/environment.sh"
exec "${MUSIC_COVER_BASE_PREFIX}/bin/python" -m music_cover.voice_training --stage 01_train "$@" --adaptation lora
