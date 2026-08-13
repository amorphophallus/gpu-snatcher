#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SOURCE="$SCRIPT_DIR/gpu_reservation_worker.py"

ACTION=""
HOST_ALIAS=""
GPU_ID=""
TARGET_RATIO="0.95"
HEADROOM_MIB="1024"
YIELD_USER="hy"
POLL_SECONDS="0.5"
MAX_START_USED_RATIO="0.10"
REMOTE_PYTHON="/mnt/nas/share/home/hy/miniconda3/envs/rr/bin/python"
REMOTE_STATE_DIR="/home/hy/.cache/gpu-snatcher"
SESSION_NAME=""
START_TIMEOUT_SECONDS="30"
DRY_RUN=0

usage() {
    cat <<'EOF'
Reserve one explicitly selected remote GPU by holding CUDA memory.

Usage:
  reserve_gpu.sh start   --host HOST --gpu ID [options]
  reserve_gpu.sh status  --host HOST --gpu ID [--session NAME]
  reserve_gpu.sh logs    --host HOST --gpu ID [--session NAME]
  reserve_gpu.sh release --host HOST --gpu ID [--session NAME]

Required:
  --host HOST                 SSH alias, for example zju_4090_240
  --gpu ID                    Physical nvidia-smi GPU index

Start options:
  --target-ratio RATIO        Target total memory use, default 0.95
  --headroom-mib MIB          Minimum memory left free, default 1024
  --yield-user USER           Auto-release for another process from USER,
                              default hy
  --poll-seconds SEC          Process detection interval, default 0.5
  --max-start-used-ratio R    Refuse a non-idle GPU, default 0.10
  --remote-python PATH        Python with CUDA PyTorch
  --remote-state-dir PATH     Remote state/log directory
  --session NAME              Reservation name; default reserve_gpu_<ID>
  --dry-run                   Validate and print without remote writes

The worker never kills another process. It releases its own tensors and exits
when another process owned by --yield-user appears on the selected GPU. For a
planned large job, run `release` before starting it so the job cannot race the
automatic detector and encounter OOM.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        start|status|logs|release)
            [[ -z "$ACTION" ]] || die "action specified more than once"
            ACTION="$1"
            shift
            ;;
        --host)
            HOST_ALIAS="${2:?missing value for --host}"
            shift 2
            ;;
        --gpu)
            GPU_ID="${2:?missing value for --gpu}"
            shift 2
            ;;
        --target-ratio)
            TARGET_RATIO="${2:?missing value for --target-ratio}"
            shift 2
            ;;
        --headroom-mib)
            HEADROOM_MIB="${2:?missing value for --headroom-mib}"
            shift 2
            ;;
        --yield-user)
            YIELD_USER="${2:?missing value for --yield-user}"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="${2:?missing value for --poll-seconds}"
            shift 2
            ;;
        --max-start-used-ratio)
            MAX_START_USED_RATIO="${2:?missing value for --max-start-used-ratio}"
            shift 2
            ;;
        --remote-python)
            REMOTE_PYTHON="${2:?missing value for --remote-python}"
            shift 2
            ;;
        --remote-state-dir)
            REMOTE_STATE_DIR="${2:?missing value for --remote-state-dir}"
            shift 2
            ;;
        --session)
            SESSION_NAME="${2:?missing value for --session}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$ACTION" ]] || die "action is required"
[[ "$HOST_ALIAS" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid --host"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ "$YIELD_USER" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] || die "invalid --yield-user"
[[ "$REMOTE_STATE_DIR" == /* ]] || die "--remote-state-dir must be absolute"
[[ "$REMOTE_PYTHON" == /* ]] || die "--remote-python must be absolute"

SESSION_NAME="${SESSION_NAME:-reserve_gpu_${GPU_ID}}"
[[ "$SESSION_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid --session"

python3 - "$TARGET_RATIO" "$HEADROOM_MIB" "$POLL_SECONDS" \
    "$MAX_START_USED_RATIO" <<'PY'
import sys

target, headroom, poll, max_used = sys.argv[1:]
target = float(target)
headroom = int(headroom)
poll = float(poll)
max_used = float(max_used)
if not 0.0 < target < 1.0:
    raise SystemExit("--target-ratio must be between zero and one")
if headroom < 256:
    raise SystemExit("--headroom-mib must be at least 256")
if poll < 0.1:
    raise SystemExit("--poll-seconds must be at least 0.1")
if not 0.0 <= max_used < 1.0:
    raise SystemExit("--max-start-used-ratio must be in [0, 1)")
PY

SSH_ARGS=(
    -o BatchMode=yes
    -o ConnectTimeout=8
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
)

remote_prefix="${REMOTE_STATE_DIR}/${SESSION_NAME}"
remote_worker="${remote_prefix}.py"
remote_pid="${remote_prefix}.pid"
remote_state="${remote_prefix}.json"
remote_log="${remote_prefix}.log"

if ((DRY_RUN)); then
    printf 'action=%s host=%s gpu=%s session=%s\n' \
        "$ACTION" "$HOST_ALIAS" "$GPU_ID" "$SESSION_NAME"
    printf 'target_ratio=%s headroom_mib=%s yield_user=%s poll_seconds=%s\n' \
        "$TARGET_RATIO" "$HEADROOM_MIB" "$YIELD_USER" "$POLL_SECONDS"
    printf 'remote_python=%s state_prefix=%s\n' "$REMOTE_PYTHON" "$remote_prefix"
    exit 0
fi

run_remote() {
    ssh "${SSH_ARGS[@]}" "$HOST_ALIAS" "$@"
}

case "$ACTION" in
    start)
        [[ -f "$WORKER_SOURCE" ]] || die "worker not found: $WORKER_SOURCE"
        worker_b64="$(base64 < "$WORKER_SOURCE" | tr -d '\r\n')"
        run_remote bash -s -- \
            "$GPU_ID" "$SESSION_NAME" "$TARGET_RATIO" "$HEADROOM_MIB" \
            "$YIELD_USER" "$POLL_SECONDS" "$MAX_START_USED_RATIO" \
            "$REMOTE_PYTHON" "$REMOTE_STATE_DIR" "$worker_b64" <<'REMOTE'
set -euo pipefail

process_is_live() {
    local pid="$1"
    local process_state
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    process_state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$process_state" && "${process_state:0:1}" != "Z" ]]
}

gpu_id="$1"
session_name="$2"
target_ratio="$3"
headroom_mib="$4"
yield_user="$5"
poll_seconds="$6"
max_start_used_ratio="$7"
remote_python="$8"
state_dir="$9"
worker_b64="${10}"

prefix="${state_dir}/${session_name}"
worker_path="${prefix}.py"
pid_path="${prefix}.pid"
state_path="${prefix}.json"
log_path="${prefix}.log"
start_lock_path="${prefix}.start.lock"

mkdir -p "$state_dir"
[[ -x "$remote_python" ]] || {
    echo "ERROR: remote Python is not executable: $remote_python" >&2
    exit 1
}
command -v flock >/dev/null || {
    echo "ERROR: flock is required on the remote host" >&2
    exit 1
}
exec 9> "$start_lock_path"
flock -n 9 || {
    echo "ERROR: another start operation is in progress for $session_name" >&2
    exit 1
}

if [[ -f "$pid_path" ]]; then
    old_pid="$(cat "$pid_path" 2>/dev/null || true)"
    if process_is_live "$old_pid"; then
        echo "ERROR: reservation $session_name is already running as PID $old_pid" >&2
        exit 1
    fi
fi

gpu_row="$(
    nvidia-smi \
        --query-gpu=index,uuid,memory.total,memory.used \
        --format=csv,noheader,nounits \
    | awk -F, -v wanted="$gpu_id" '
        { gsub(/^[ \t]+|[ \t]+$/, "", $1) }
        $1 == wanted { print; found=1 }
        END { if (!found) exit 1 }
    '
)" || {
    echo "ERROR: GPU $gpu_id was not returned by nvidia-smi" >&2
    exit 1
}

IFS=, read -r observed_index gpu_uuid total_mib used_mib <<< "$gpu_row"
gpu_uuid="${gpu_uuid//[[:space:]]/}"
total_mib="${total_mib//[[:space:]]/}"
used_mib="${used_mib//[[:space:]]/}"

existing_pids="$(
    nvidia-smi \
        --query-compute-apps=gpu_uuid,pid \
        --format=csv,noheader,nounits \
    | awk -F, -v wanted="$gpu_uuid" '
        {
            gsub(/^[ \t]+|[ \t]+$/, "", $1)
            gsub(/^[ \t]+|[ \t]+$/, "", $2)
        }
        $1 == wanted { print $2 }
    '
)"
if [[ -n "$existing_pids" ]]; then
    echo "ERROR: GPU $gpu_id already has compute PID(s): ${existing_pids//$'\n'/,}" >&2
    exit 1
fi

python3 - "$used_mib" "$total_mib" "$max_start_used_ratio" <<'PY'
import sys

used, total, maximum = map(float, sys.argv[1:])
ratio = used / total
if ratio >= maximum:
    raise SystemExit(
        f"GPU is not idle enough to reserve: {used:.0f}/{total:.0f} MiB "
        f"({ratio:.1%}) >= {maximum:.1%}"
    )
PY

printf '%s' "$worker_b64" | base64 -d > "$worker_path"
chmod 700 "$worker_path"
rm -f "$state_path"

nohup env CUDA_VISIBLE_DEVICES="$gpu_id" \
    "$remote_python" "$worker_path" \
    --physical-gpu "$gpu_id" \
    --gpu-uuid "$gpu_uuid" \
    --target-ratio "$target_ratio" \
    --headroom-mib "$headroom_mib" \
    --yield-user "$yield_user" \
    --poll-seconds "$poll_seconds" \
    --state-path "$state_path" \
    > "$log_path" 2>&1 < /dev/null &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$pid_path"
echo "started reservation $session_name on GPU $gpu_id ($gpu_uuid), PID $worker_pid"
REMOTE

        deadline=$((SECONDS + START_TIMEOUT_SECONDS))
        while ((SECONDS < deadline)); do
            state_output="$(run_remote bash -s -- "$remote_pid" "$remote_state" <<'REMOTE'
set -euo pipefail
process_is_live() {
    local pid="$1"
    local process_state
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    process_state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$process_state" && "${process_state:0:1}" != "Z" ]]
}
pid_path="$1"
state_path="$2"
pid="$(cat "$pid_path" 2>/dev/null || true)"
if [[ -f "$state_path" ]]; then
    cat "$state_path"
elif process_is_live "$pid"; then
    printf '{"state":"starting","pid":%s}\n' "$pid"
else
    printf '{"state":"missing"}\n'
fi
REMOTE
)"
            worker_state="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("state", "missing"))' <<< "$state_output")"
            case "$worker_state" in
                holding)
                    printf '%s\n' "$state_output"
                    exit 0
                    ;;
                error|released|missing)
                    printf '%s\n' "$state_output" >&2
                    run_remote tail -n 80 "$remote_log" >&2 || true
                    exit 1
                    ;;
            esac
            sleep 1
        done
        die "reservation did not become ready within ${START_TIMEOUT_SECONDS}s"
        ;;
    status)
        run_remote bash -s -- "$GPU_ID" "$remote_pid" "$remote_state" <<'REMOTE'
set -euo pipefail
process_is_live() {
    local pid="$1"
    local process_state
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    process_state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$process_state" && "${process_state:0:1}" != "Z" ]]
}
gpu_id="$1"
pid_path="$2"
state_path="$3"
pid="$(cat "$pid_path" 2>/dev/null || true)"
if [[ -f "$state_path" ]]; then
    cat "$state_path"
else
    echo '{"state":"never_started"}'
fi
if process_is_live "$pid"; then
    echo "process_alive=true"
else
    echo "process_alive=false"
fi
nvidia-smi -i "$gpu_id" \
    --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
REMOTE
        ;;
    logs)
        run_remote tail -n 100 "$remote_log"
        ;;
    release)
        run_remote bash -s -- "$remote_pid" "$remote_worker" "$remote_state" <<'REMOTE'
set -euo pipefail
process_is_live() {
    local pid="$1"
    local process_state
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    process_state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$process_state" && "${process_state:0:1}" != "Z" ]]
}
pid_path="$1"
worker_path="$2"
state_path="$3"
pid="$(cat "$pid_path" 2>/dev/null || true)"
if ! process_is_live "$pid"; then
    echo "reservation is not running"
    exit 0
fi
cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
case "$cmdline" in
    *"$worker_path"*) ;;
    *)
        echo "ERROR: refusing to signal PID $pid; command does not match worker" >&2
        exit 1
        ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 50); do
    if ! process_is_live "$pid"; then
        [[ -f "$state_path" ]] && cat "$state_path"
        exit 0
    fi
    sleep 0.2
done
echo "ERROR: worker did not exit after SIGTERM; no SIGKILL was sent" >&2
exit 1
REMOTE
        ;;
esac
