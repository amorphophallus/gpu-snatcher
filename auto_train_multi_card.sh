#!/usr/bin/env bash

set -euo pipefail

join_command_parts() {
    python3 - "$@" <<'PY'
import shlex
import sys

print(shlex.join(sys.argv[1:]))
PY
}

# Multi-card training command.
TRAIN_COMMAND_PARTS=(
    torchrun
    --standalone
    --nproc_per_node=2
    -m
    src.train.bc_ddp
    +experiment=rgbd/diff_unet
    "task=[one_leg,round_table,lamp]"
    data.demo_source=rollout
    data.data_subset=500
    data.demo_outcome=success
    data.suffix=rgbd-skill
    training.batch_size=256
    training.num_epochs=3000
    training.steps_per_epoch=-1
    training.save_per_epoch=500
    wandb.project=multi-task-rgbd-skill-low-500
    wandb.mode=online
    randomness=low
    dryrun=false
    training.num_epochs=4000
)
TRAIN_COMMAND="$(join_command_parts "${TRAIN_COMMAND_PARTS[@]}")"
SSH_NAME="230"
NUM_GPUS="2"
GPU_ID="0,1"
DATA_DIR_PROCESSED="/data/hy/robust-rearrangement-custom/data/"
FAST_SERVER=(236 230)
SLOW_SERVER=(228 238 240)

SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
MEMORY_USAGE_THRESHOLD="${MEMORY_USAGE_THRESHOLD:-0.1}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-5}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-5}"
POLL_TIMEOUT_SECONDS="${POLL_TIMEOUT_SECONDS:-300}"
SSH_COMMAND_TIMEOUT_SECONDS="${SSH_COMMAND_TIMEOUT_SECONDS:-15}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/mnt/nas/share/home/hy/robust-rearrangement-custom}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-rr}"

SESSION_NAME_CANDIDATES=(
    atlas
    birch
    cedar
    comet
    delta
    ember
    lotus
    maple
    nova
    pine
    river
    stone
)

get_hosts_from_ssh_config() {
    if [[ ! -f "$SSH_CONFIG_PATH" ]]; then
        echo "SSH config not found: $SSH_CONFIG_PATH" >&2
        return 1
    fi

    awk '
        BEGIN { IGNORECASE = 1 }
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*Host[[:space:]]+/ {
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^zju_4090_/ && $i !~ /[*?]/) {
                    print $i
                }
            }
        }
    ' "$SSH_CONFIG_PATH" | awk '!seen[$0]++'
}

invoke_ssh() {
    local host_alias="$1"
    local remote_command="$2"

    if command -v timeout >/dev/null 2>&1; then
        timeout "${SSH_COMMAND_TIMEOUT_SECONDS}s" \
            ssh \
            -o BatchMode=yes \
            -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
            -o ServerAliveInterval=5 \
            -o ServerAliveCountMax=1 \
            "$host_alias" \
            "$remote_command"
    else
        ssh \
            -o BatchMode=yes \
            -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
            -o ServerAliveInterval=5 \
            -o ServerAliveCountMax=1 \
            "$host_alias" \
            "$remote_command"
    fi
}

get_host_gpu_status() {
    local host_alias="$1"
    local query_output
    local ssh_status

    query_output="$(ssh -o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
        -o StrictHostKeyChecking=accept-new \
        "$host_alias" \
        "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>&1)"
    ssh_status=$?

    if [[ $ssh_status -ne 0 ]]; then
        printf 'HOST|%s|DOWN|%s\n' "$host_alias" "$query_output"
        return 0
    fi

    printf 'HOST|%s|OK|\n' "$host_alias"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        python3 - "$host_alias" "$MEMORY_USAGE_THRESHOLD" "$line" <<'PY'
import sys

host = sys.argv[1]
threshold = float(sys.argv[2])
line = sys.argv[3]
parts = [p.strip() for p in line.split(",")]
if len(parts) < 4:
    sys.exit(0)

index = int(parts[0])
memory_total = float(parts[1])
memory_used = float(parts[2])
gpu_util = float(parts[3])
usage_ratio = memory_used / memory_total if memory_total > 0 else 1.0
usage_percent = round(usage_ratio * 100, 1)
status = "FREE" if usage_ratio < threshold else "BUSY"

print(
    f"GPU|{host}|{index}|{status}|{int(round(memory_used))}|{int(round(memory_total))}|{usage_percent}|{int(round(gpu_util))}"
)
PY
    done <<< "$query_output"
}

normalize_gpu_id_list() {
    python3 - "$1" <<'PY'
import sys

text = sys.argv[1]
if not text.strip():
    print("")
    raise SystemExit(0)

seen = set()
normalized = []
for raw_part in text.split(","):
    part = raw_part.strip()
    if not part:
        raise SystemExit("GPU_ID must be a comma-separated list of non-negative integers without empty items.")
    if not part.isdigit():
        raise SystemExit(f"GPU_ID entries must be non-negative integers, got '{part}'.")

    value = int(part)
    if value in seen:
        raise SystemExit(f"GPU_ID contains a duplicate entry: {value}.")

    seen.add(value)
    normalized.append(str(value))

print(",".join(normalized))
PY
}

list_hosts_by_priority() {
    local fast_csv
    local slow_csv

    fast_csv="$(IFS=,; echo "${FAST_SERVER[*]}")"
    slow_csv="$(IFS=,; echo "${SLOW_SERVER[*]}")"

    get_hosts_from_ssh_config | python3 - "$fast_csv" "$slow_csv" <<'PY'
import re
import sys

fast = {item for item in sys.argv[1].split(",") if item}
slow = {item for item in sys.argv[2].split(",") if item}
hosts = [line.strip() for line in sys.stdin if line.strip()]

def sort_key(host):
    match = re.search(r"(\d+)$", host)
    suffix = match.group(1) if match else ""
    host_num = int(suffix) if suffix else -1
    priority = 1
    if suffix in fast:
        priority = 0
    elif suffix in slow:
        priority = 2
    return (priority, -host_num, host)

for host in sorted(dict.fromkeys(hosts), key=sort_key):
    print(host)
PY
}

select_gpus_on_host() {
    local host_alias="$1"
    local num_gpus="$2"
    local preferred_gpu_csv="${3:-}"

    python3 - "$host_alias" "$num_gpus" "$preferred_gpu_csv" <<'PY'
import sys

host_alias = sys.argv[1]
num_gpus = int(sys.argv[2])
preferred_gpu_csv = sys.argv[3]

host_state = "DOWN"
host_note = ""
reported_gpu_ids = []
free_gpus = {}

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue

    parts = line.split("|")
    row_type = parts[0]
    if row_type == "HOST":
        host_state = parts[2]
        host_note = parts[3] if len(parts) > 3 else ""
        continue

    if row_type != "GPU":
        continue

    gpu_id = int(parts[2])
    reported_gpu_ids.append(gpu_id)
    if parts[3] != "FREE":
        continue

    free_gpus[gpu_id] = {
        "id": gpu_id,
        "used": float(parts[4]),
        "util": float(parts[7]),
    }

if host_state != "OK":
    print(f"DOWN|{host_alias}|{host_note}")
    raise SystemExit(0)

preferred_gpu_ids = []
if preferred_gpu_csv.strip():
    preferred_gpu_ids = [int(item) for item in preferred_gpu_csv.split(",") if item]

selected_gpu_ids = []
if preferred_gpu_ids:
    for gpu_id in preferred_gpu_ids:
        if gpu_id in free_gpus:
            selected_gpu_ids.append(gpu_id)
            if len(selected_gpu_ids) == num_gpus:
                break

if len(selected_gpu_ids) < num_gpus:
    sorted_free_gpus = sorted(free_gpus.values(), key=lambda item: (item["util"], item["used"], item["id"]))
    selected_gpu_ids = [item["id"] for item in sorted_free_gpus[:num_gpus]]

if len(selected_gpu_ids) < num_gpus:
    print(f"INSUFFICIENT|{host_alias}|{len(free_gpus)}|{num_gpus}")
    raise SystemExit(0)

gpu_ids_csv = ",".join(str(gpu_id) for gpu_id in selected_gpu_ids)
gpu_utils_csv = ",".join(str(int(round(free_gpus[gpu_id]["util"]))) for gpu_id in selected_gpu_ids)
print(f"OK|{host_alias}|{gpu_ids_csv}|{gpu_utils_csv}|{len(free_gpus)}")
PY
}

find_multi_gpu_target_or_error() {
    local ssh_name="$1"
    local num_gpus="$2"
    local preferred_gpu_csv="$3"
    local host_alias
    local selection_result
    local status
    local field2
    local field3

    if [[ ! "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_GPUS must be a positive integer, got '$num_gpus'." >&2
        return 1
    fi

    if [[ -n "${ssh_name// }" ]]; then
        host_alias="zju_4090_${ssh_name}"
        selection_result="$(get_host_gpu_status "$host_alias" | select_gpus_on_host "$host_alias" "$num_gpus" "$preferred_gpu_csv")"
        IFS='|' read -r status _ field2 field3 _ <<< "$selection_result"

        case "$status" in
            OK)
                printf '%s\n' "$selection_result"
                return 0
                ;;
            DOWN)
                echo "Preferred host '$host_alias' is unreachable: $field2" >&2
                return 1
                ;;
            INSUFFICIENT)
                echo "Host '$host_alias' has only $field2 free GPUs; need $field3." >&2
                return 1
                ;;
            *)
                echo "Failed to select GPUs on host '$host_alias'." >&2
                return 1
                ;;
        esac
    fi

    while IFS= read -r host_alias; do
        [[ -z "$host_alias" ]] && continue
        selection_result="$(get_host_gpu_status "$host_alias" | select_gpus_on_host "$host_alias" "$num_gpus" "")"
        IFS='|' read -r status _ _ _ _ <<< "$selection_result"
        if [[ "$status" == "OK" ]]; then
            printf '%s\n' "$selection_result"
            return 0
        fi
    done < <(list_hosts_by_priority)

    echo "No reachable server has ${num_gpus} free GPUs." >&2
    return 1
}

prepare_train_command() {
    python3 - "$TRAIN_COMMAND" "$1" "$2" <<'PY'
import os
import re
import shlex
import sys

command = sys.argv[1].strip()
gpu_ids_csv = sys.argv[2]
num_gpus = sys.argv[3]

if not command:
    raise SystemExit("TRAIN_COMMAND is empty.")

parts = shlex.split(command)
if not parts:
    raise SystemExit("TRAIN_COMMAND is empty.")

env_parts = []
env_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
index = 0
while index < len(parts) and env_pattern.match(parts[index]) and not parts[index].startswith(("/", "./")):
    key = parts[index].split("=", 1)[0]
    if key != "CUDA_VISIBLE_DEVICES":
        env_parts.append(parts[index])
    index += 1

command_parts = parts[index:]
if not command_parts or os.path.basename(command_parts[0]) != "torchrun":
    raise SystemExit("TRAIN_COMMAND must start with torchrun for auto_train_multi_card.")

filtered_parts = []
cursor = 0
while cursor < len(command_parts):
    token = command_parts[cursor]
    if re.match(r"^training\.gpu_id=\S+$", token):
        cursor += 1
        continue
    if token == "training.gpu_id":
        cursor += 2 if cursor + 1 < len(command_parts) else 1
        continue
    if token == "--nproc_per_node":
        cursor += 2 if cursor + 1 < len(command_parts) else 1
        continue
    if token.startswith("--nproc_per_node="):
        cursor += 1
        continue
    filtered_parts.append(token)
    cursor += 1

updated_parts = [filtered_parts[0], f"--nproc_per_node={num_gpus}", *filtered_parts[1:]]
final_parts = [f"CUDA_VISIBLE_DEVICES={gpu_ids_csv}", *env_parts, *updated_parts]
print(shlex.join(final_parts))
PY
}

get_command_name() {
    python3 - "$1" <<'PY'
import os
import shlex
import sys

command = sys.argv[1].strip()
if not command:
    print("unknown")
    raise SystemExit(0)

parts = shlex.split(command)
while parts and "=" in parts[0] and not parts[0].startswith(("/", "./")):
    key = parts[0].split("=", 1)[0]
    if not key.replace("_", "a").isalnum():
        break
    parts = parts[1:]

if not parts:
    print("unknown")
elif os.path.basename(parts[0]).startswith("python") and len(parts) > 1:
    if parts[1] == "-m" and len(parts) > 2:
        print(parts[2])
    elif not parts[1].startswith("-"):
        print(os.path.basename(parts[1]))
    else:
        print(os.path.basename(parts[0]))
else:
    print(os.path.basename(parts[0]))
PY
}

pick_tmux_session_name() {
    local host_alias="$1"
    local candidate

    for candidate in "${SESSION_NAME_CANDIDATES[@]}"; do
        if invoke_ssh "$host_alias" "tmux has-session -t '$candidate' >/dev/null 2>&1"; then
            continue
        fi

        echo "$candidate"
        return 0
    done

    echo "No available tmux session name in candidate list." >&2
    return 1
}

start_remote_training() {
    local host_alias="$1"
    local session_name="$2"
    local prepared_command="$3"
    local encoded_command

    encoded_command="$(printf '%s' "$prepared_command" | base64 -w 0)"

    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
        "$host_alias" \
        bash -s -- "$session_name" "$REMOTE_PROJECT_DIR" "$REMOTE_CONDA_ENV" "$encoded_command" "$DATA_DIR_PROCESSED" <<'REMOTE'
set -euo pipefail

session_name="$1"
project_dir="$2"
conda_env="$3"
encoded_train_command="$4"
data_dir_processed="${5:-}"
train_command="$(printf '%s' "$encoded_train_command" | base64 -d)"

command -v tmux >/dev/null 2>&1
tmux has-session -t "$session_name" >/dev/null 2>&1 && exit 10
tmux new-session -d -s "$session_name"
tmux set-option -t "$session_name" remain-on-exit on
tmux new-window -t "$session_name" -n train
tmux send-keys -t "$session_name:train" -l "cd $project_dir"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "source ~/.bashrc >/dev/null 2>&1 || true"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l 'eval "$(conda shell.bash hook 2>/dev/null)" || true'
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "export TMPDIR=/tmp TEMP=/tmp TMP=/tmp"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "conda activate $conda_env"
tmux send-keys -t "$session_name:train" Enter
if [[ -n "${data_dir_processed// }" ]]; then
    printf -v data_dir_export 'export DATA_DIR_PROCESSED=%q' "$data_dir_processed"
    tmux send-keys -t "$session_name:train" -l "$data_dir_export"
    tmux send-keys -t "$session_name:train" Enter
fi
tmux send-keys -t "$session_name:train" -l "echo __AUTO_TRAIN_READY__"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "$train_command"
tmux send-keys -t "$session_name:train" Enter
tmux kill-window -t "${session_name}:0" >/dev/null 2>&1 || true
REMOTE
}

capture_tmux_output() {
    local host_alias="$1"
    local session_name="$2"

    invoke_ssh "$host_alias" \
        "tmux capture-pane -pt $(printf '%q' "$session_name:train") -S -200"
}

extract_wandb_run_name() {
    python3 - <<'PY'
import re
import sys

text = sys.stdin.read()
text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
patterns = [
    r'wandb run name\s*[:=]\s*(.+)',
    r'wandb[: ]+run name\s*[:=]\s*(.+)',
]

for line in text.splitlines():
    stripped = line.strip()
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            print(match.group(1).strip())
            raise SystemExit(0)

raise SystemExit(1)
PY
}

extract_failure_reason() {
    python3 - <<'PY'
import re
import sys

text = sys.stdin.read()
text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
patterns = [
    r'^Traceback \(most recent call last\):.*',
    r'^.*Error executing job with overrides:.*',
    r'^.*FileNotFoundError:.*',
    r'^.*ModuleNotFoundError:.*',
    r'^.*RuntimeError:.*',
    r'^.*OSError:.*',
    r'^.*AssertionError:.*',
    r'^.*UnboundLocalError:.*',
    r'^.*ValueError:.*',
    r'^.*KeyError:.*',
    r'^.*IndexError:.*',
    r'^.*TypeError:.*',
    r'^.*No space left on device.*',
    r'^.*command not found.*',
    r'^.*Killed$',
]

matches = []
for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    for pattern in patterns:
        if re.search(pattern, line, flags=re.IGNORECASE):
            if line not in matches:
                matches.append(line)
            break

if matches:
    print(" | ".join(matches))
    raise SystemExit(0)

raise SystemExit(1)
PY
}

write_structured_status() {
    local status="$1"
    local server="$2"
    local num_gpus="$3"
    local gpu_ids="$4"
    local tmux_name="$5"
    local command_name="$6"
    local wandb_run_name="$7"
    local error_reason="${8:-}"

    printf 'status: %s\n' "$status"
    printf 'server: %s\n' "$server"
    printf 'num_gpus: %s\n' "$num_gpus"
    printf 'gpu_ids: %s\n' "$gpu_ids"
    printf 'tmux_name: %s\n' "$tmux_name"
    printf 'command_name: %s\n' "$command_name"
    printf 'wandb_run_name: %s\n' "$wandb_run_name"
    if [[ -n "${error_reason// }" ]]; then
        printf 'error_reason: %s\n' "$error_reason"
    fi
}

main() {
    if [[ -z "${TRAIN_COMMAND// }" ]]; then
        echo "Set TRAIN_COMMAND at the top of this script before running it." >&2
        exit 1
    fi

    local normalized_gpu_id_csv
    if ! normalized_gpu_id_csv="$(normalize_gpu_id_list "$GPU_ID")"; then
        exit 1
    fi

    local selected
    if ! selected="$(find_multi_gpu_target_or_error "$SSH_NAME" "$NUM_GPUS" "$normalized_gpu_id_csv")"; then
        exit 1
    fi

    local selection_status host_alias gpu_ids_csv
    IFS='|' read -r selection_status host_alias gpu_ids_csv _ _ <<< "$selected"
    local prepared_command
    if ! prepared_command="$(prepare_train_command "$gpu_ids_csv" "$NUM_GPUS")"; then
        exit 1
    fi

    local command_name
    command_name="$(get_command_name "$prepared_command")"
    local session_name
    session_name="$(pick_tmux_session_name "$host_alias")"

    if ! start_remote_training "$host_alias" "$session_name" "$prepared_command"; then
        echo "Failed to start tmux session '$session_name' on $host_alias." >&2
        exit 1
    fi

    local start_time
    start_time="$(date +%s)"
    local wandb_run_name=""
    local failure_reason=""
    local pane_output=""

    while true; do
        if (( $(date +%s) - start_time >= POLL_TIMEOUT_SECONDS )); then
            write_structured_status "timeout" "$host_alias" "$NUM_GPUS" "$gpu_ids_csv" "$session_name" "$command_name" "-" "Timed out waiting for wandb run name"
            exit 1
        fi

        sleep "$POLL_INTERVAL_SECONDS"

        if (( $(date +%s) - start_time >= POLL_TIMEOUT_SECONDS )); then
            write_structured_status "timeout" "$host_alias" "$NUM_GPUS" "$gpu_ids_csv" "$session_name" "$command_name" "-" "Timed out waiting for wandb run name"
            exit 1
        fi

        if ! pane_output="$(capture_tmux_output "$host_alias" "$session_name" 2>&1)"; then
            echo "Failed to capture tmux output from $host_alias:$session_name" >&2
            echo "$pane_output" >&2
            exit 1
        fi

        if wandb_run_name="$(printf '%s' "$pane_output" | extract_wandb_run_name 2>/dev/null)"; then
            break
        fi

        if failure_reason="$(printf '%s' "$pane_output" | extract_failure_reason 2>/dev/null)"; then
            write_structured_status "failed" "$host_alias" "$NUM_GPUS" "$gpu_ids_csv" "$session_name" "$command_name" "-" "$failure_reason"
            exit 1
        fi

        if (( $(date +%s) - start_time >= POLL_TIMEOUT_SECONDS )); then
            write_structured_status "timeout" "$host_alias" "$NUM_GPUS" "$gpu_ids_csv" "$session_name" "$command_name" "-" "Timed out waiting for wandb run name"
            exit 1
        fi
    done

    write_structured_status "started" "$host_alias" "$NUM_GPUS" "$gpu_ids_csv" "$session_name" "$command_name" "$wandb_run_name"
}

main "$@"
