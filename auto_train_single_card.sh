#!/usr/bin/env bash

set -euo pipefail

join_command_parts() {
    python3 - "$@" <<'PY'
import shlex
import sys

print(shlex.join(sys.argv[1:]))
PY
}

get_command_part_value() {
    local key="$1"
    shift
    local part

    for part in "$@"; do
        if [[ "$part" == "$key="* ]]; then
            printf '%s\n' "${part#"$key="}"
            return 0
        fi
    done

    return 1
}

DATA_STORAGE_FORMAT="lmdb"
DATA_LOAD_INTO_MEMORY="false"
DATA_PATHS_OVERRIDE=""

# Single-card training command.
TRAIN_COMMAND_PARTS=(
    python
    -m
    src.train.bc
    +experiment=rgbd/diff_unet
    "task=[one_leg,round_table,lamp]"
    data.demo_source=rollout
    data.demo_outcome=success
    data.suffix=rgbd-skill
    "data.storage_format=${DATA_STORAGE_FORMAT}"
    "data.load_into_memory=${DATA_LOAD_INTO_MEMORY}"
    data.data_subset=50
    training.batch_size=256
    training.num_epochs=5000
    training.steps_per_epoch=-1
    training.save_per_epoch=1000
    wandb.project=multi-task-rgbd-skill-low-smalldot
    wandb.mode=online
    training.gpu_id=0
    randomness=low
    dryrun=false
)

# Optional explicit dataset override.
if [[ -n "${DATA_PATHS_OVERRIDE// }" ]]; then
    TRAIN_COMMAND_PARTS+=("data.data_paths_override=${DATA_PATHS_OVERRIDE}")
fi
TRAIN_COMMAND="$(join_command_parts "${TRAIN_COMMAND_PARTS[@]}")"
WANDB_PROJECT_NAME="$(get_command_part_value wandb.project "${TRAIN_COMMAND_PARTS[@]}" || printf 'project')"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-project}"
SSH_NAME="230"
GPU_ID="0"
DATA_DIR_PROCESSED="/data/hy/robust-rearrangement-custom/data/"  # server local
DATA_DIR_PROCESSED="~/robust-rearrangement-custom/data/"  # home, for 236
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
    local restore_errexit=0

    # Match check_zju_4090.sh semantics under `set -e`: failed SSH probes should
    # report a DOWN row with the error message instead of exiting early.
    case $- in
        *e*)
            restore_errexit=1
            set +e
            ;;
    esac

    query_output="$(ssh -o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
        -o StrictHostKeyChecking=accept-new \
        "$host_alias" \
        "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>&1)"
    ssh_status=$?

    if (( restore_errexit )); then
        set -e
    fi

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

find_first_free_gpu() {
    local host_alias
    local line
    local candidates=()
    local fast_csv slow_csv
    fast_csv="$(IFS=,; echo "${FAST_SERVER[*]}")"
    slow_csv="$(IFS=,; echo "${SLOW_SERVER[*]}")"

    while IFS= read -r host_alias; do
        [[ -z "$host_alias" ]] && continue
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            IFS='|' read -r row_type row_host field1 field2 field3 _field4 _field5 field6 <<< "$line"
            if [[ "$row_type" == "GPU" && "$field2" == "FREE" ]]; then
                candidates+=("${row_host}|${field1}|${field6}|${field3}")
            fi
        done < <(get_host_gpu_status "$host_alias")
    done < <(get_hosts_from_ssh_config)

    if [[ ${#candidates[@]} -gt 0 ]]; then
        printf '%s\n' "${candidates[@]}" |
            awk -F'|' -v fast_csv="$fast_csv" -v slow_csv="$slow_csv" '
                BEGIN {
                    split(fast_csv, fast_arr, ",")
                    split(slow_csv, slow_arr, ",")
                    for (i in fast_arr) if (fast_arr[i] != "") fast_map[fast_arr[i]] = 1
                    for (i in slow_arr) if (slow_arr[i] != "") slow_map[slow_arr[i]] = 1
                }
                {
                    host_num = 0
                    suffix = ""
                    if (match($1, /[0-9]+$/)) {
                        suffix = substr($1, RSTART, RLENGTH)
                        host_num = suffix + 0
                    }

                    priority = 1
                    if (suffix in fast_map) {
                        priority = 0
                    } else if (suffix in slow_map) {
                        priority = 2
                    }

                    print $0 "|" priority "|" host_num
                }
            ' |
            sort -t'|' -k3,3n -k4,4n -k5,5n -k6,6nr -k1,1 -k2,2n |
            head -n 1 |
            cut -d'|' -f1-4
        return 0
    fi

    return 1
}

find_preferred_gpu_or_error() {
    local ssh_name="$1"
    local gpu_id_text="$2"
    local host_alias="zju_4090_${ssh_name}"
    local selected
    local host_state="DOWN"
    local host_note=""
    local line
    local -a seen_gpu_ids=()

    if [[ ! "$gpu_id_text" =~ ^[0-9]+$ ]]; then
        echo "GPU_ID must be a non-negative integer, got '$gpu_id_text'." >&2
        return 1
    fi

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        IFS='|' read -r row_type row_host field1 field2 field3 _field4 field5 field6 <<< "$line"
        if [[ "$row_type" == "HOST" ]]; then
            host_state="$field1"
            host_note="$field2"
            continue
        fi

        if [[ "$row_type" != "GPU" ]]; then
            continue
        fi

        seen_gpu_ids+=("$field1")
        if [[ "$field1" != "$gpu_id_text" ]]; then
            continue
        fi

        if [[ "$field2" != "FREE" ]]; then
            echo "Preferred GPU not available: ${host_alias} GPU${gpu_id_text} memory usage ${field5}% >= threshold $(python3 - "$MEMORY_USAGE_THRESHOLD" <<'PY'
import sys
print(round(float(sys.argv[1]) * 100, 1))
PY
)% (gpu util ${field6}%)." >&2
            return 1
        fi

        selected="${row_host}|${field1}|${field6}|${field3}"
    done < <(get_host_gpu_status "$host_alias")

    if [[ "$host_state" != "OK" ]]; then
        echo "Preferred host '$host_alias' is unreachable: $host_note" >&2
        return 1
    fi

    if [[ -z "$selected" ]]; then
        local available_gpu_ids
        if [[ ${#seen_gpu_ids[@]} -gt 0 ]]; then
            available_gpu_ids="$(IFS=', '; printf '%s' "${seen_gpu_ids[*]}")"
        else
            available_gpu_ids="none"
        fi
        echo "Preferred GPU not found on host '$host_alias': GPU${gpu_id_text}. Available GPU IDs reported by nvidia-smi: ${available_gpu_ids}. GPU IDs are 0-based." >&2
        return 1
    fi

    printf '%s\n' "$selected"
}

prepare_train_command() {
    python3 - "$TRAIN_COMMAND" "$1" <<'PY'
import re
import sys

command = sys.argv[1].strip()
gpu_id = sys.argv[2]

if not command:
    raise SystemExit("TRAIN_COMMAND is empty.")

if re.search(r'(^|\s)training\.gpu_id=\S+', command):
    updated = re.sub(r'(^|\s)training\.gpu_id=\S+', rf'\1training.gpu_id={gpu_id}', command, count=1)
else:
    updated = f"{command} training.gpu_id={gpu_id}"

print(updated)
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
        bash -s -- "$session_name" "$REMOTE_PROJECT_DIR" "$REMOTE_CONDA_ENV" "$encoded_command" "$DATA_DIR_PROCESSED" "$WANDB_PROJECT_NAME" <<'REMOTE'
set -euo pipefail

session_name="$1"
project_dir="$2"
conda_env="$3"
encoded_train_command="$4"
data_dir_processed="${5:-}"
wandb_project_name="${6:-project}"
train_command="$(printf '%s' "$encoded_train_command" | base64 -d)"

expand_path() {
    local path="$1"
    if [[ "$path" == "~" ]]; then
        printf '%s\n' "$HOME"
    elif [[ "$path" == "~/"* ]]; then
        printf '%s\n' "$HOME/${path#"~/"}"
    else
        printf '%s\n' "$path"
    fi
}

project_dir="$(expand_path "$project_dir")"
data_dir_processed="$(expand_path "$data_dir_processed")"

command -v tmux >/dev/null 2>&1
tmux has-session -t "$session_name" >/dev/null 2>&1 && exit 10
tmux new-session -d -s "$session_name"
tmux set-option -t "$session_name" remain-on-exit on
tmux new-window -t "$session_name" -n train
printf -v project_cd_command 'cd %q' "$project_dir"
tmux send-keys -t "$session_name:train" -l "$project_cd_command"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "source ~/.bashrc >/dev/null 2>&1 || true"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l 'eval "$(conda shell.bash hook 2>/dev/null)" || true'
tmux send-keys -t "$session_name:train" Enter
wandb_project_slug="$(printf '%s' "${wandb_project_name:-project}" | tr -c 'A-Za-z0-9._-' '_')"
if [[ -z "$wandb_project_slug" ]]; then
    wandb_project_slug="project"
fi
runtime_tmp_dir="/tmp/wandb-${wandb_project_slug}"
wandb_cache_dir="${runtime_tmp_dir}/cache"
wandb_config_dir="${runtime_tmp_dir}/config"
wandb_data_dir="${runtime_tmp_dir}/data"
wandb_artifact_dir="${runtime_tmp_dir}/artifacts"
mkdir -p "$runtime_tmp_dir" "$wandb_cache_dir" "$wandb_config_dir" "$wandb_data_dir" "$wandb_artifact_dir"
printf -v runtime_env_export 'export TMPDIR=%q TEMP=%q TMP=%q WANDB_DIR=%q WANDB_CACHE_DIR=%q WANDB_CONFIG_DIR=%q WANDB_DATA_DIR=%q WANDB_ARTIFACT_DIR=%q' "$runtime_tmp_dir" "$runtime_tmp_dir" "$runtime_tmp_dir" "$runtime_tmp_dir" "$wandb_cache_dir" "$wandb_config_dir" "$wandb_data_dir" "$wandb_artifact_dir"
tmux send-keys -t "$session_name:train" -l "$runtime_env_export"
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

main() {
    if [[ -z "${TRAIN_COMMAND// }" ]]; then
        echo "Set TRAIN_COMMAND at the top of this script before running it." >&2
        exit 1
    fi

    local selected
    if [[ -n "${SSH_NAME// }" && -n "${GPU_ID// }" ]]; then
        if ! selected="$(find_preferred_gpu_or_error "$SSH_NAME" "$GPU_ID")"; then
            exit 1
        fi
    else
        if ! selected="$(find_first_free_gpu)"; then
            echo "No reachable free GPU found." >&2
            exit 1
        fi
    fi

    local host_alias gpu_id gpu_util
    IFS='|' read -r host_alias gpu_id gpu_util _ <<< "$selected"
    local prepared_command
    prepared_command="$(prepare_train_command "$gpu_id")"
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
            printf 'status: timeout\n'
            printf 'server: %s\n' "$host_alias"
            printf 'gpu_id: %s\n' "$gpu_id"
            printf 'gpu_util: %s\n' "$(printf '%.0f' "$gpu_util")"
            printf 'tmux_name: %s\n' "$session_name"
            printf 'command_name: %s\n' "$command_name"
            printf 'wandb_run_name: -\n'
            printf 'error_reason: Timed out waiting for wandb run name\n'
            exit 1
        fi

        sleep "$POLL_INTERVAL_SECONDS"

        if (( $(date +%s) - start_time >= POLL_TIMEOUT_SECONDS )); then
            printf 'status: timeout\n'
            printf 'server: %s\n' "$host_alias"
            printf 'gpu_id: %s\n' "$gpu_id"
            printf 'gpu_util: %s\n' "$(printf '%.0f' "$gpu_util")"
            printf 'tmux_name: %s\n' "$session_name"
            printf 'command_name: %s\n' "$command_name"
            printf 'wandb_run_name: -\n'
            printf 'error_reason: Timed out waiting for wandb run name\n'
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
            printf 'status: failed\n'
            printf 'server: %s\n' "$host_alias"
            printf 'gpu_id: %s\n' "$gpu_id"
            printf 'gpu_util: %s\n' "$(printf '%.0f' "$gpu_util")"
            printf 'tmux_name: %s\n' "$session_name"
            printf 'command_name: %s\n' "$command_name"
            printf 'wandb_run_name: -\n'
            printf 'error_reason: %s\n' "$failure_reason"
            exit 1
        fi

        if (( $(date +%s) - start_time >= POLL_TIMEOUT_SECONDS )); then
            printf 'status: timeout\n'
            printf 'server: %s\n' "$host_alias"
            printf 'gpu_id: %s\n' "$gpu_id"
            printf 'gpu_util: %s\n' "$(printf '%.0f' "$gpu_util")"
            printf 'tmux_name: %s\n' "$session_name"
            printf 'command_name: %s\n' "$command_name"
            printf 'wandb_run_name: -\n'
            printf 'error_reason: Timed out waiting for wandb run name\n'
            exit 1
        fi
    done

    printf 'status: started\n'
    printf 'server: %s\n' "$host_alias"
    printf 'gpu_id: %s\n' "$gpu_id"
    printf 'gpu_util: %s\n' "$(printf '%.0f' "$gpu_util")"
    printf 'tmux_name: %s\n' "$session_name"
    printf 'command_name: %s\n' "$command_name"
    printf 'wandb_run_name: %s\n' "$wandb_run_name"
}

main "$@"
