#!/usr/bin/env bash

set -euo pipefail

TRAIN_COMMAND=""

SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
MEMORY_USAGE_THRESHOLD="${MEMORY_USAGE_THRESHOLD:-0.1}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-5}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-5}"
POLL_TIMEOUT_SECONDS="${POLL_TIMEOUT_SECONDS:-900}"
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

    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
        "$host_alias" \
        "$remote_command"
}

find_first_free_gpu() {
    local host_alias
    local output
    local line
    local candidate
    local candidates=()

    while IFS= read -r host_alias; do
        [[ -z "$host_alias" ]] && continue
        if ! output="$(invoke_ssh "$host_alias" \
            "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>&1)"; then
            continue
        fi

        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            if candidate="$(python3 - "$host_alias" "$MEMORY_USAGE_THRESHOLD" "$line" <<'PY'
import sys

host = sys.argv[1]
threshold = float(sys.argv[2])
line = sys.argv[3]
parts = [p.strip() for p in line.split(",")]
if len(parts) < 4:
    sys.exit(0)

gpu_id = int(parts[0])
memory_total = float(parts[1])
memory_used = float(parts[2])
gpu_util = float(parts[3])
usage_ratio = memory_used / memory_total if memory_total > 0 else 1.0

if usage_ratio < threshold:
    print(f"{host}|{gpu_id}|{gpu_util}|{memory_used}")
    sys.exit(0)
sys.exit(1)
PY
            )"; then
                candidates+=("$candidate")
            fi
        done <<< "$output"
    done < <(get_hosts_from_ssh_config)

    if [[ ${#candidates[@]} -gt 0 ]]; then
        printf '%s\n' "${candidates[@]}" | sort -t'|' -k3,3n -k4,4n -k1,1 -k2,2n | head -n 1
        return 0
    fi

    return 1
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
        bash -s -- "$session_name" "$REMOTE_PROJECT_DIR" "$REMOTE_CONDA_ENV" "$encoded_command" <<'REMOTE'
set -euo pipefail

session_name="$1"
project_dir="$2"
conda_env="$3"
encoded_train_command="$4"
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
    if ! selected="$(find_first_free_gpu)"; then
        echo "No reachable free GPU found." >&2
        exit 1
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
        sleep "$POLL_INTERVAL_SECONDS"
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
