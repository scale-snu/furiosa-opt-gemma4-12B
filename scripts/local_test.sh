#!/bin/bash
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"

FIXTURE="ref/fixtures.safetensors"

for argument in "$@"; do
    case "$argument" in
        -h|--help) echo '사용법: ./scripts/local_test.sh — 로컬 RNGD에서 Stage 1 테스트 실행'; exit 0 ;;
        *) echo "local_test.sh: unknown argument $argument" >&2; exit 2 ;;
    esac
done

if [ ! -f "$FIXTURE" ]; then
    echo "local_test.sh: $FIXTURE is missing -- generate it first:" >&2
    echo "    python3 scripts/generate_references.py" >&2
    exit 1
fi

echo "==> running test_kernels"
TUC_PROFILE_LEVEL="${TUC_PROFILE_LEVEL:-info}" "$CRATE/scripts/furiosa.sh" test --release --test test_kernels
