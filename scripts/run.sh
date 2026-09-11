#!/bin/bash
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"

"$CRATE/scripts/furiosa.sh" test --release
"$CRATE/scripts/furiosa.sh" run --release --bin gemma4 -- "$@"
