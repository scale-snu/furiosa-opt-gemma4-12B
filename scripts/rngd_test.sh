#!/bin/bash
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"
export PATH="$CRATE/target/toolchains/furiosa-opt-0.6.0/bin:$PATH"

FIXTURE="ref/fixtures.safetensors"
POLL_SECONDS="${RNGD_POLL_SECONDS:-5}"
WAIT_TIMEOUT="${RNGD_WAIT_TIMEOUT:-1800}"

build=1
wait_for_result=1
for argument in "$@"; do
    case "$argument" in
        --no-build) build=0 ;;
        --no-wait) wait_for_result=0 ;;
        -h|--help)
            cat <<'USAGE'
Usage: ./scripts/rngd_test.sh [--no-build] [--no-wait]

Build and submit the Stage 1 kernel tests with furiosa-arena.
Run `furiosa-arena login` once before submitting.

  --no-build  Reuse the latest test binary (only if sources are unchanged).
  --no-wait   Return after submission and print the server's numeric job ID.

Environment:
  FURIOSA_ARENA_URL  Controller URL (CLI default when omitted).
  RNGD_URL          Legacy controller URL, used when FURIOSA_ARENA_URL is unset.
  RNGD_TIMEOUT      Optional remote execution limit in seconds (server default).
  RNGD_WAIT_TIMEOUT Local wait limit including queue time (default: 1800 seconds).
  RNGD_POLL_SECONDS Status polling interval (default: 5 seconds).
  RNGD_JOB_NAME     Optional job name; this is not the server's numeric job ID.
USAGE
            exit 0 ;;
        *) echo "rngd_test.sh: unknown argument $argument" >&2; exit 2 ;;
    esac
done

if ! command -v furiosa-arena >/dev/null 2>&1; then
    echo "rngd_test.sh: furiosa-arena is missing; install it with: cargo binstall furiosa-arena-cli" >&2
    exit 127
fi

if [[ ! "$WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "rngd_test.sh: RNGD_WAIT_TIMEOUT must be a positive integer in seconds" >&2
    exit 2
fi
timeout_args=()
if [ -n "${RNGD_TIMEOUT:-}" ]; then
    if [[ ! "$RNGD_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
        echo "rngd_test.sh: RNGD_TIMEOUT must be a positive integer in seconds" >&2
        exit 2
    fi
    timeout_args=(--timeout "$RNGD_TIMEOUT")
fi

if [ -z "${FURIOSA_ARENA_URL:-}" ] && [ -z "${RNGD_URL:-}" ] && [ -f "$HOME/.bashrc" ]; then
    bashrc_line=$(grep -E '^[[:space:]]*export[[:space:]]+RNGD_URL=' "$HOME/.bashrc" | tail -1 || true)
    if [ -n "$bashrc_line" ]; then
        bashrc_value=${bashrc_line#*=}
        bashrc_value=${bashrc_value%\"}; bashrc_value=${bashrc_value#\"}
        bashrc_value=${bashrc_value%\'}; bashrc_value=${bashrc_value#\'}
        export RNGD_URL="$bashrc_value"
    fi
fi

if [ -z "${FURIOSA_ARENA_URL:-}" ] && [ -n "${RNGD_URL:-}" ]; then
    export FURIOSA_ARENA_URL="$RNGD_URL"
fi

find_test_binary() {
    find target/release/deps -maxdepth 1 -type f -name 'test_kernels-*' ! -name '*.d' -perm -u+x \
        2>/dev/null | xargs -r ls -t | head -1
}

if [ "$build" -eq 1 ]; then
    echo "==> building test_kernels (as a cargo test binary)"
    build_json=$(cargo furiosa-opt test --release --test test_kernels --no-run --message-format=json-render-diagnostics)
    BINARY=$(printf '%s\n' "$build_json" \
        | grep '"kind":\["test"\]' \
        | grep '"name":"test_kernels"' \
        | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' \
        | tail -1)
    if [ -z "$BINARY" ]; then
        echo "rngd_test.sh: could not find the built test_kernels binary in cargo's build output" >&2
        exit 1
    fi
else
    BINARY="$(find_test_binary)"
fi

if [ ! -f "$FIXTURE" ]; then
    echo "rngd_test.sh: $FIXTURE is missing -- generate it first:" >&2
    echo "    python3 scripts/generate_references.py" >&2
    exit 1
fi
for required in "$BINARY" scripts/rngd/remote_entrypoint.sh; do
    if [ -z "$required" ] || [ ! -f "$required" ]; then
        echo "rngd_test.sh: test_kernels binary or scripts/rngd/remote_entrypoint.sh is missing -- run without --no-build" >&2
        exit 1
    fi
done

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp scripts/rngd/remote_entrypoint.sh "$staging/remote_entrypoint.sh"
cp "$BINARY" "$staging/test_runtime"
cp "$FIXTURE" "$staging/fixtures.safetensors"
chmod +x "$staging/remote_entrypoint.sh" "$staging/test_runtime"

job_name="${RNGD_JOB_NAME:-rngd_test_$RANDOM}"

echo "==> submitting $job_name ($(du -ch "$staging"/* | tail -1 | cut -f1) total)"
submit_output=$(furiosa-arena submit \
    "$staging/remote_entrypoint.sh" \
    "$staging/test_runtime" \
    "$staging/fixtures.safetensors" \
    --name "$job_name" \
    --entrypoint remote_entrypoint.sh \
    "${timeout_args[@]}" 2>&1); then
    printf '%s\n' "$submit_output"
else
    submit_status=$?
    printf '%s\n' "$submit_output" >&2
    echo "rngd_test.sh: submission failed (exit $submit_status); no job ID was confirmed" >&2
    exit "$submit_status"
fi

job=$(printf '%s\n' "$submit_output" | sed -n 's/.*submitted job \([0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$job" ]; then
    echo "rngd_test.sh: could not find a job ID in furiosa-arena's output; check furiosa-arena list before retrying" >&2
    exit 1
fi

echo "==> server job ID: $job (name: $job_name); check with: furiosa-arena status $job"

if [ "$wait_for_result" -eq 0 ]; then
    echo "==> submitted job $job; follow it with: furiosa-arena logs $job --follow"
    exit 0
fi

json_field() {
    sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\1/p" <<<"$1" | head -1
}

is_terminal() {
    case "$1" in
        succeeded|failed|completed|cancelled|canceled) return 0 ;;
        *) return 1 ;;
    esac
}

echo "==> waiting on job $job (polling every ${POLL_SECONDS}s)"
deadline=$(( SECONDS + WAIT_TIMEOUT ))
state=""
status_output=""
while [ "$SECONDS" -lt "$deadline" ]; do
    status_output=$(furiosa-arena status "$job" 2>&1 || true)
    status_output=$(furiosa-arena status "$job" 2>&1 || true)
    state=$(json_field "$status_output" status | tr '[:upper:]' '[:lower:]')
    is_terminal "$state" && break
    case "$state" in
        queued|running) ;;
        *) echo "rngd_test.sh: unexpected status '${state:-unreadable}', retrying" >&2 ;;
    esac
    sleep "$POLL_SECONDS"
done

if ! is_terminal "$state"; then
    echo "rngd_test.sh: job $job still '${state:-unknown}' after ${WAIT_TIMEOUT}s; cancel with: furiosa-arena cancel $job" >&2
    exit 1
fi

code=$(json_field "$status_output" exit_code)
echo "==> job $job $state (exit ${code:-?}); log follows"
furiosa-arena logs "$job" || true
furiosa-arena logs "$job" || true

[ "${code:-1}" = "0" ]
