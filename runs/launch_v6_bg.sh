#!/usr/bin/env bash
# =============================================================================
# Launch runs/train_a800_x8_v6.sh in the background with logs streamed to a
# timestamped file. Survives ssh disconnects without tmux.
#
# Usage:
#   bash runs/launch_v6_bg.sh              # start from SKIP_TO=1 (default)
#   bash runs/launch_v6_bg.sh 3            # start from SKIP_TO=3
#   bash runs/launch_v6_bg.sh 7            # start from RL only
#
#   bash runs/launch_v6_bg.sh status       # PID + GPU + last 5 log lines
#   bash runs/launch_v6_bg.sh tail         # tail -f the active log
#   bash runs/launch_v6_bg.sh stop         # kill the run
#
# Any extra env (FORCE_MERGE=1, RL_NUM_SAMPLES=32, etc.) passes through:
#   FORCE_MERGE=1 bash runs/launch_v6_bg.sh 3
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/runs/logs"
PID_FILE="${LOG_DIR}/v6.pid"
LOG_LINK="${LOG_DIR}/v6_latest.log"
TRAIN_SCRIPT="${REPO_ROOT}/runs/train_a800_x8_v6.sh"

mkdir -p "${LOG_DIR}"

cmd=${1:-start}

# If $1 is a number, it's SKIP_TO and implies start
case "$cmd" in
    ''|start|[0-9]*)
        if [[ "$cmd" =~ ^[0-9]+$ ]]; then
            SKIP_TO="$cmd"
        else
            SKIP_TO="${SKIP_TO:-1}"
        fi
        cmd=start
        ;;
esac

is_running() {
    [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

case "$cmd" in

    start)
        if is_running; then
            echo "ERROR: already running (PID $(cat "${PID_FILE}"))" >&2
            echo "       stop first: bash runs/launch_v6_bg.sh stop" >&2
            exit 1
        fi
        if [ ! -f "${TRAIN_SCRIPT}" ]; then
            echo "ERROR: ${TRAIN_SCRIPT} not found" >&2
            exit 1
        fi

        TS="$(date +%Y%m%d_%H%M%S)"
        LOG="${LOG_DIR}/v6_${TS}.log"

        # Preserve all inbound env (FORCE_MERGE, RL_*, etc.) by running in a
        # subshell; only SKIP_TO is set from the CLI arg.
        nohup bash -c "
            cd '${REPO_ROOT}'
            export SKIP_TO='${SKIP_TO}'
            echo '=== launched $(date) — SKIP_TO=${SKIP_TO} ==='
            bash '${TRAIN_SCRIPT}'
            echo '=== exited $(date) with code \$? ==='
        " > "${LOG}" 2>&1 < /dev/null &

        PID=$!
        echo "$PID" > "${PID_FILE}"
        ln -sf "$(basename "${LOG}")" "${LOG_LINK}"

        echo "started v6 training"
        echo "  PID   = ${PID}"
        echo "  LOG   = ${LOG}"
        echo "  alias = ${LOG_LINK}"
        echo "  SKIP_TO=${SKIP_TO}"
        echo ""
        echo "follow:  tail -f ${LOG_LINK}"
        echo "status:  bash runs/launch_v6_bg.sh status"
        echo "stop:    bash runs/launch_v6_bg.sh stop"
        ;;

    status)
        if ! is_running; then
            echo "not running"
            if [ -L "${LOG_LINK}" ]; then
                echo ""
                echo "last log: $(readlink -f "${LOG_LINK}")"
                echo "--- tail ---"
                tail -n 10 "${LOG_LINK}" 2>/dev/null || true
            fi
            exit 1
        fi
        PID="$(cat "${PID_FILE}")"
        echo "running  PID=${PID}"
        ps -p "${PID}" -o pid,etime,pcpu,pmem,cmd | tail -n +1
        echo ""
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "--- GPU ---"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                       --format=csv,noheader
            echo ""
        fi
        if [ -L "${LOG_LINK}" ]; then
            echo "--- log tail (${LOG_LINK}) ---"
            tail -n 10 "${LOG_LINK}"
        fi
        ;;

    tail|log|follow)
        if [ ! -L "${LOG_LINK}" ] && [ ! -f "${LOG_LINK}" ]; then
            echo "no log yet" >&2
            exit 1
        fi
        exec tail -f "${LOG_LINK}"
        ;;

    stop|kill)
        if ! is_running; then
            echo "not running"
            exit 0
        fi
        PID="$(cat "${PID_FILE}")"
        echo "stopping PID ${PID} ..."
        kill "${PID}" 2>/dev/null || true
        # Kids (torchrun → 8 workers) are in the same pgid. Give them a
        # chance, then force.
        sleep 3
        if kill -0 "${PID}" 2>/dev/null; then
            echo "  still alive, SIGKILL"
            kill -9 "${PID}" 2>/dev/null || true
        fi
        # Mop up orphaned torchrun workers even if parent already exited.
        if pgrep -f "torch.distributed.run.*scripts.chat_(sft|rl_funcall)" >/dev/null 2>&1; then
            echo "  killing orphaned torchrun workers"
            pkill -9 -f "torch.distributed.run.*scripts.chat_(sft|rl_funcall)" || true
        fi
        rm -f "${PID_FILE}"
        echo "stopped"
        ;;

    *)
        echo "usage: bash runs/launch_v6_bg.sh [start|<skip_to>|status|tail|stop]" >&2
        exit 2
        ;;
esac
