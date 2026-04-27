#!/usr/bin/env bash

set -euo pipefail

# Comment out a line to skip that step.
STEPS=(
    download
    eval
)

REMOTE_PATH="/mnt/nas/share/home/hy/robust-rearrangement-custom/"
REMOTE_SSH_HOST="230"
RUN_ID="exalted-meadow-11"
LOCAL_PATH="~/projects/robust-rearrangement-custom"
TASK="round_table"
PROJECT="rgbd_skill"
MODEL_ARCH="diff_unet"
NUM_DATA="200"
EPOCH=""
N_ENVS=3
N_ROLLOUTS=18
VISUALIZE=false
DEBUG=false
CONDA_ENV="rr"
CHECKPOINT_PATTERN="*last*.pt"  # 暂时使用 last
CONNECT_TIMEOUT_SECONDS=10

# Optional CLI override. If empty, it is derived from the local checkpoint filename (without extension).
ROLLOUT_SUFFIX_MODEL_NAME=""

PARAMS=(
    --if-exists append
    --max-rollout-steps 1000
    --action-type pos
    --observation-space image
    --randomness low
    --save-rollouts
    --save-failures
    --save-depth-image
    --annotate-skill
    --skill-on-image
)

log_info() {
    printf '[%s] INFO %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log_error() {
    printf '[%s] ERROR %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
    log_error "$*"
    exit 1
}

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

require_command() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

checkpoint_pattern_to_name_tag() {
    local pattern="$1"
    local tag

    tag="$(basename -- "$pattern")"
    tag="${tag%.pt}"
    tag="${tag//\*/}"
    tag="${tag//\?/}"
    tag="${tag//\[/}"
    tag="${tag//\]/}"

    [[ -n "${tag// }" ]] || die "Unable to derive checkpoint name tag from CHECKPOINT_PATTERN: ${pattern}"
    printf '%s\n' "$tag"
}

quote_command() {
    local quoted=""
    local arg
    for arg in "$@"; do
        printf -v quoted '%s%q ' "$quoted" "$arg"
    done
    printf '%s\n' "${quoted% }"
}

normalize_remote_ssh_host() {
    local host="$1"
    host="${host//[[:space:]]/}"

    if [[ -z "$host" ]]; then
        printf '%s\n' ""
    elif [[ "$host" =~ ^[0-9]+$ ]]; then
        printf 'zju_4090_%s\n' "$host"
    else
        printf '%s\n' "$host"
    fi
}

discover_checkpoint() {
    local outputs_root="$1"
    local run_id="$2"
    local checkpoint_pattern="$3"
    local epoch="${4:-}"
    local ssh_host="${5:-}"
    local result

    if [[ -n "$ssh_host" ]]; then
        local remote_cmd
        printf -v remote_cmd 'python3 - %q %q %q %q' "$outputs_root" "$run_id" "$checkpoint_pattern" "$epoch"
        result="$(
            ssh \
                -o BatchMode=yes \
                -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
                "$ssh_host" \
                "$remote_cmd" <<'PY'
import fnmatch
import os
import sys
from datetime import datetime
from pathlib import Path

outputs_root = Path(sys.argv[1])
run_id = sys.argv[2]
checkpoint_pattern = sys.argv[3]
epoch = sys.argv[4].strip()


def parse_output_timestamp(date_text: str, time_text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H-%M-%S.%f", "%Y-%m-%d %H-%M-%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported output timestamp: {date_text}/{time_text}")


if not outputs_root.is_dir():
    raise SystemExit(f"Outputs directory not found: {outputs_root}")

output_dir_map = {}
for root, dirs, _files in os.walk(outputs_root):
    for dirname in dirs:
        full_path = Path(root) / dirname
        if run_id not in str(full_path):
            continue
        parts = full_path.parts
        try:
            outputs_index = parts.index("outputs")
        except ValueError:
            continue
        if len(parts) <= outputs_index + 2:
            continue
        date_text = parts[outputs_index + 1]
        time_text = parts[outputs_index + 2]
        try:
            timestamp = parse_output_timestamp(date_text, time_text)
        except ValueError:
            continue
        output_dir = Path(*parts[: outputs_index + 3])
        output_dir_map[str(output_dir)] = timestamp

if not output_dir_map:
    raise SystemExit(f"No run directories containing RUN_ID '{run_id}' found under {outputs_root}")

output_dir_candidates = sorted(
    ((timestamp, output_dir_text) for output_dir_text, timestamp in output_dir_map.items()),
    key=lambda item: (item[0], item[1]),
)


def has_epoch_suffix(checkpoint: Path, epoch_text: str) -> bool:
    return checkpoint.stem.endswith(f"_{epoch_text}")


def find_checkpoint_in_output(output_dir: Path):
    models_dir = output_dir / "models"
    if not models_dir.is_dir():
        return None

    checkpoint_candidates = []
    for checkpoint in models_dir.rglob("*.pt"):
        if not checkpoint.is_file():
            continue
        if not fnmatch.fnmatch(checkpoint.name, checkpoint_pattern):
            continue
        if epoch:
            if not has_epoch_suffix(checkpoint, epoch):
                continue
        checkpoint_candidates.append((checkpoint.stat().st_mtime, str(checkpoint)))

    if not checkpoint_candidates:
        return None

    _mtime, checkpoint_text = max(checkpoint_candidates, key=lambda item: (item[0], item[1]))
    return checkpoint_text


selected_output_dir_text = None
selected_checkpoint_text = None

if epoch:
    for _timestamp, output_dir_text in reversed(output_dir_candidates):
        checkpoint_text = find_checkpoint_in_output(Path(output_dir_text))
        if checkpoint_text is None:
            continue
        selected_output_dir_text = output_dir_text
        selected_checkpoint_text = checkpoint_text
        break

    if selected_checkpoint_text is None:
        raise SystemExit(
            f"No checkpoints matching pattern '{checkpoint_pattern}' with required suffix "
            f"'_{epoch}' found under outputs for RUN_ID '{run_id}'"
        )
else:
    _timestamp, output_dir_text = output_dir_candidates[-1]
    selected_output_dir_text = output_dir_text
    selected_checkpoint_text = find_checkpoint_in_output(Path(output_dir_text))
    if selected_checkpoint_text is None:
        raise SystemExit(
            f"No checkpoints matching pattern '{checkpoint_pattern}' found under latest output: "
            f"{Path(output_dir_text) / 'models'}"
        )

print(selected_output_dir_text)
print(selected_checkpoint_text)
PY
        )"
    else
        result="$(
            python3 - "$outputs_root" "$run_id" "$checkpoint_pattern" "$epoch" <<'PY'
import fnmatch
import os
import sys
from datetime import datetime
from pathlib import Path

outputs_root = Path(sys.argv[1])
run_id = sys.argv[2]
checkpoint_pattern = sys.argv[3]
epoch = sys.argv[4].strip()


def parse_output_timestamp(date_text: str, time_text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H-%M-%S.%f", "%Y-%m-%d %H-%M-%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported output timestamp: {date_text}/{time_text}")


if not outputs_root.is_dir():
    raise SystemExit(f"Outputs directory not found: {outputs_root}")

output_dir_map = {}
for root, dirs, _files in os.walk(outputs_root):
    for dirname in dirs:
        full_path = Path(root) / dirname
        if run_id not in str(full_path):
            continue
        parts = full_path.parts
        try:
            outputs_index = parts.index("outputs")
        except ValueError:
            continue
        if len(parts) <= outputs_index + 2:
            continue
        date_text = parts[outputs_index + 1]
        time_text = parts[outputs_index + 2]
        try:
            timestamp = parse_output_timestamp(date_text, time_text)
        except ValueError:
            continue
        output_dir = Path(*parts[: outputs_index + 3])
        output_dir_map[str(output_dir)] = timestamp

if not output_dir_map:
    raise SystemExit(f"No run directories containing RUN_ID '{run_id}' found under {outputs_root}")

output_dir_candidates = sorted(
    ((timestamp, output_dir_text) for output_dir_text, timestamp in output_dir_map.items()),
    key=lambda item: (item[0], item[1]),
)


def has_epoch_suffix(checkpoint: Path, epoch_text: str) -> bool:
    return checkpoint.stem.endswith(f"_{epoch_text}")


def find_checkpoint_in_output(output_dir: Path):
    models_dir = output_dir / "models"
    if not models_dir.is_dir():
        return None

    checkpoint_candidates = []
    for checkpoint in models_dir.rglob("*.pt"):
        if not checkpoint.is_file():
            continue
        if not fnmatch.fnmatch(checkpoint.name, checkpoint_pattern):
            continue
        if epoch:
            if not has_epoch_suffix(checkpoint, epoch):
                continue
        checkpoint_candidates.append((checkpoint.stat().st_mtime, str(checkpoint)))

    if not checkpoint_candidates:
        return None

    _mtime, checkpoint_text = max(checkpoint_candidates, key=lambda item: (item[0], item[1]))
    return checkpoint_text


selected_output_dir_text = None
selected_checkpoint_text = None

if epoch:
    for _timestamp, output_dir_text in reversed(output_dir_candidates):
        checkpoint_text = find_checkpoint_in_output(Path(output_dir_text))
        if checkpoint_text is None:
            continue
        selected_output_dir_text = output_dir_text
        selected_checkpoint_text = checkpoint_text
        break

    if selected_checkpoint_text is None:
        raise SystemExit(
            f"No checkpoints matching pattern '{checkpoint_pattern}' with required suffix "
            f"'_{epoch}' found under outputs for RUN_ID '{run_id}'"
        )
else:
    _timestamp, output_dir_text = output_dir_candidates[-1]
    selected_output_dir_text = output_dir_text
    selected_checkpoint_text = find_checkpoint_in_output(Path(output_dir_text))
    if selected_checkpoint_text is None:
        raise SystemExit(
            f"No checkpoints matching pattern '{checkpoint_pattern}' found under latest output: "
            f"{Path(output_dir_text) / 'models'}"
        )

print(selected_output_dir_text)
print(selected_checkpoint_text)
PY
        )"
    fi

    printf '%s\n' "$result"
}

copy_checkpoint_to_local() {
    local remote_checkpoint="$1"
    local local_checkpoint="$2"
    local ssh_host="${3:-}"

    mkdir -p "$(dirname "$local_checkpoint")"

    if [[ -n "$ssh_host" ]]; then
        require_command scp
        log_info "Downloading checkpoint via scp from ${ssh_host}:${remote_checkpoint}"
        scp \
            -o BatchMode=yes \
            -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
            "${ssh_host}:${remote_checkpoint}" \
            "$local_checkpoint"
    else
        [[ -f "$remote_checkpoint" ]] || die "Checkpoint not found on local filesystem: $remote_checkpoint"
        log_info "Copying checkpoint from locally mounted path: $remote_checkpoint"
        cp "$remote_checkpoint" "$local_checkpoint"
    fi
}

activate_conda_env() {
    local env_name="$1"

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook 2>/dev/null)"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        die "Unable to initialize conda. Please install conda or update activate_conda_env()."
    fi

    conda activate "$env_name"
}

download_checkpoint_step() {
    local local_checkpoint="$1"
    local remote_root outputs_root remote_ssh_host
    local output_dir checkpoint_source discovery_result
    local -a discovery_lines

    require_command python3

    remote_root="${REMOTE_PATH%/}"
    outputs_root="${remote_root}/outputs"
    remote_ssh_host="$(normalize_remote_ssh_host "${REMOTE_SSH_HOST:-}")"

    log_info "Starting checkpoint download for RUN_ID=${RUN_ID}"
    if [[ -n "$remote_ssh_host" ]]; then
        require_command ssh
        log_info "Using remote SSH host: ${remote_ssh_host}"
    else
        log_info "Using locally accessible REMOTE_PATH: ${remote_root}"
        [[ -d "$outputs_root" ]] || die "REMOTE_PATH is not available locally and REMOTE_SSH_HOST is empty: ${outputs_root}"
    fi

    if [[ -n "${EPOCH// }" ]]; then
        log_info "Searching matching output directories under ${outputs_root} for checkpoint suffix _${EPOCH}"
    else
        log_info "Searching latest output directory under ${outputs_root}"
    fi
    discovery_result="$(discover_checkpoint "$outputs_root" "$RUN_ID" "$CHECKPOINT_PATTERN" "$EPOCH" "$remote_ssh_host")"
    mapfile -t discovery_lines < <(printf '%s\n' "$discovery_result")
    [[ ${#discovery_lines[@]} -ge 2 ]] || die "Failed to parse checkpoint discovery result."
    output_dir="${discovery_lines[0]}"
    checkpoint_source="${discovery_lines[1]}"

    log_info "Selected output directory: ${output_dir}"
    log_info "Selected checkpoint: ${checkpoint_source}"
    log_info "Preparing local checkpoint destination: ${local_checkpoint}"
    copy_checkpoint_to_local "$checkpoint_source" "$local_checkpoint" "$remote_ssh_host"
    log_info "Checkpoint ready at ${local_checkpoint}"
}

eval_step() {
    local local_root="$1"
    local local_checkpoint="$2"
    local rollout_suffix_model_name="$3"
    local normalized_task
    local -a task_args
    local -a eval_cmd

    [[ -f "$local_checkpoint" ]] || die "Local checkpoint not found: ${local_checkpoint}. Enable the download step or prepare the file manually."

    normalized_task="${TASK// /}"
    IFS='+' read -r -a task_args <<< "$normalized_task"
    [[ ${#task_args[@]} -gt 0 ]] || die "TASK cannot be empty."

    eval_cmd=(
        python -m src.eval.evaluate_model
        --n-envs "$N_ENVS"
        --n-rollouts "$N_ROLLOUTS"
        -f "${task_args[@]}"
        "${PARAMS[@]}"
        --rollout-suffix-model-name "$rollout_suffix_model_name"
        --wt-path "$local_checkpoint"
    )
    if [[ "$VISUALIZE" == "true" ]]; then
        eval_cmd+=(--visualize)
    fi
    if [[ "$DEBUG" == "true" ]]; then
        eval_cmd+=(--debug)
    fi

    log_info "Running evaluation in ${local_root}"
    log_info "Evaluation command: $(quote_command "${eval_cmd[@]}")"
    (
        cd "$local_root"
        activate_conda_env "$CONDA_ENV"
        "${eval_cmd[@]}"
    )
    log_info "Evaluation finished successfully."
}

print_usage() {
    cat <<'USAGE'
Usage: auto_eval.sh [--rollout-suffix-model-name NAME]

Options:
  --rollout-suffix-model-name NAME  Value passed to evaluator; defaults to local checkpoint filename stem.
  -h, --help                        Show this help.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --rollout-suffix-model-name)
                shift
                [[ $# -gt 0 ]] || die "--rollout-suffix-model-name requires a value"
                ROLLOUT_SUFFIX_MODEL_NAME="$1"
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1 (use --help)"
                ;;
        esac
    done
}

main() {
    local local_root destination_dir local_checkpoint
    local checkpoint_name_tag
    local rollout_suffix_model_name
    local step

    parse_args "$@"

    local_root="$(expand_path "$LOCAL_PATH")"
    [[ -d "$local_root" ]] || die "LOCAL_PATH does not exist: ${local_root}"

    checkpoint_name_tag="$(checkpoint_pattern_to_name_tag "$CHECKPOINT_PATTERN")"
    destination_dir="${local_root}/checkpoints/bc/${TASK}/low"
    local_checkpoint="${destination_dir}/${PROJECT}_${MODEL_ARCH}_${NUM_DATA}traj_${checkpoint_name_tag}_${EPOCH}.pt"

    rollout_suffix_model_name="$ROLLOUT_SUFFIX_MODEL_NAME"
    if [[ -z "$rollout_suffix_model_name" ]]; then
        rollout_suffix_model_name="$(basename -- "$local_checkpoint")"
        rollout_suffix_model_name="${rollout_suffix_model_name%.*}"
    fi

    if [[ ${#STEPS[@]} -eq 0 ]]; then
        log_info "No steps selected in STEPS; nothing to do."
        return 0
    fi

    for step in "${STEPS[@]}"; do
        case "$step" in
            download)
                download_checkpoint_step "$local_checkpoint"
                ;;
            eval)
                eval_step "$local_root" "$local_checkpoint" "$rollout_suffix_model_name"
                ;;
            *)
                die "Unknown step in STEPS: ${step}"
                ;;
        esac
    done
}

main "$@"
