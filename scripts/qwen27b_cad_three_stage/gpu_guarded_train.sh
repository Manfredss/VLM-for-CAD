#!/usr/bin/env bash
# Run train_27b.sh while enforcing the Nautilus GPU-utilization policy.
# A GPU is considered active when either memory occupancy or compute utilization
# reaches the configured threshold.  If every signal remains below the threshold
# after the startup grace period, the container exits so Kubernetes releases it.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-both}"

GPU_GUARD_MIN_PERCENT="${GPU_GUARD_MIN_PERCENT:-50}"
GPU_GUARD_GRACE_SECONDS="${GPU_GUARD_GRACE_SECONDS:-90}"
GPU_GUARD_LOW_SECONDS="${GPU_GUARD_LOW_SECONDS:-120}"
GPU_GUARD_INTERVAL_SECONDS="${GPU_GUARD_INTERVAL_SECONDS:-30}"

for value_name in \
    GPU_GUARD_MIN_PERCENT \
    GPU_GUARD_GRACE_SECONDS \
    GPU_GUARD_LOW_SECONDS \
    GPU_GUARD_INTERVAL_SECONDS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${value_name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done
if (( GPU_GUARD_INTERVAL_SECONDS == 0 || GPU_GUARD_LOW_SECONDS == 0 )); then
    echo "ERROR: GPU guard interval and low-duration must be positive." >&2
    exit 2
fi

echo ">>> GPU guard: active when memory OR utilization >= ${GPU_GUARD_MIN_PERCENT}%"
echo ">>> GPU guard: grace=${GPU_GUARD_GRACE_SECONDS}s, continuous-low limit=${GPU_GUARD_LOW_SECONDS}s, interval=${GPU_GUARD_INTERVAL_SECONDS}s"

bash "${SCRIPT_DIR}/train_27b.sh" "${ACTION}" &
TRAIN_PID=$!
STARTED_AT="$(date +%s)"
LOW_SECONDS=0
QUERY_FAILURE_SECONDS=0

terminate_child() {
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap 'terminate_child; exit 143' TERM INT

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${GPU_GUARD_INTERVAL_SECONDS}"
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        break
    fi

    NOW="$(date +%s)"
    ELAPSED=$((NOW - STARTED_AT))
    METRICS="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${METRICS}" ]]; then
        QUERY_FAILURE_SECONDS=$((QUERY_FAILURE_SECONDS + GPU_GUARD_INTERVAL_SECONDS))
        echo ">>> GPU guard: elapsed=${ELAPSED}s nvidia-smi query failed (${QUERY_FAILURE_SECONDS}s continuous)"
        if (( ELAPSED >= GPU_GUARD_GRACE_SECONDS && QUERY_FAILURE_SECONDS >= GPU_GUARD_LOW_SECONDS )); then
            echo "ERROR: GPU telemetry unavailable for too long; exiting to release the allocation." >&2
            terminate_child
            exit 42
        fi
        continue
    fi
    QUERY_FAILURE_SECONDS=0

    ALL_LOW=true
    DISPLAY=()
    while IFS=',' read -r USED TOTAL UTIL; do
        USED="${USED//[[:space:]]/}"
        TOTAL="${TOTAL//[[:space:]]/}"
        UTIL="${UTIL//[[:space:]]/}"
        if [[ ! "${USED}" =~ ^[0-9]+$ || ! "${TOTAL}" =~ ^[0-9]+$ || ! "${UTIL}" =~ ^[0-9]+$ || "${TOTAL}" == "0" ]]; then
            echo "ERROR: malformed nvidia-smi metrics: used=${USED}, total=${TOTAL}, util=${UTIL}" >&2
            terminate_child
            exit 42
        fi
        MEM_PERCENT=$((100 * USED / TOTAL))
        DISPLAY+=("mem=${USED}/${TOTAL}MiB(${MEM_PERCENT}%) util=${UTIL}%")
        if (( MEM_PERCENT >= GPU_GUARD_MIN_PERCENT || UTIL >= GPU_GUARD_MIN_PERCENT )); then
            ALL_LOW=false
        fi
    done <<< "${METRICS}"

    echo ">>> GPU guard: elapsed=${ELAPSED}s ${DISPLAY[*]}"
    if (( ELAPSED < GPU_GUARD_GRACE_SECONDS )); then
        LOW_SECONDS=0
    elif [[ "${ALL_LOW}" == "true" ]]; then
        LOW_SECONDS=$((LOW_SECONDS + GPU_GUARD_INTERVAL_SECONDS))
        if (( LOW_SECONDS >= GPU_GUARD_LOW_SECONDS )); then
            echo "ERROR: all visible GPUs stayed below ${GPU_GUARD_MIN_PERCENT}% memory and utilization for ${LOW_SECONDS}s; exiting to release the allocation." >&2
            terminate_child
            exit 42
        fi
    else
        LOW_SECONDS=0
    fi
done

set +e
wait "${TRAIN_PID}"
STATUS=$?
set -e
echo ">>> guarded training exited with status ${STATUS}"
exit "${STATUS}"
