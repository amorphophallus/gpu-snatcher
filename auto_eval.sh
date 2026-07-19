#!/usr/bin/env bash

set -euo pipefail

# Comment out a line to skip that step.
STEPS=(
    eval
)

REMOTE_PATH="/mnt/nas/share/home/hy/robust-rearrangement-custom/"
DATA_DIR_RAW="${DATA_DIR_RAW:-/mnt/nas/share/home/hy/robust-rearrangement-custom/data}"
REMOTE_SSH_HOST="228"
RUN_ID="fiery-snowball-4"
#LOCAL_PATH="/data/hy/robust-rearrangement"  # 218
LOCAL_PATH="~/projects/robust-rearrangement-custom"  # base
TASK="round_table"  # 多任务用 + 号隔开
PROJECT="multi-task-rgbd-skill-low-0428"
MODEL_ARCH="dit"
NUM_DATA="100"
EPOCH=""
N_ENVS=3
N_ROLLOUTS=12
RANDOMNESS="med"
MAX_SAVED_ROLLOUTS=0
VISUALIZE=false
DEBUG=false
CONDA_ENV="rr"
CHECKPOINT_PATTERN="*last*.pt"
GPU_ID=0
CONNECT_TIMEOUT_SECONDS=10
SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
EVAL_ANNOTATE_SKILL=true
EVAL_GUIDANCE_POINT_ON_IMAGE=false
EVAL_GRASP_ANNOTATION_ON_IMAGE=false
EVAL_GRASP_PART_ANNOTATE=false
EVAL_SKILL_ON_IMAGE=false
EVAL_GUIDANCE_POINT_COLORED=false
EVAL_GRASP_ANNOTATION_COLORED=false
EVAL_PERTURB_MODE="none"  # none, random_small, short_large, place_slowdown
EVAL_ANNOTATION_NOISE_POS_STD_M="0"
EVAL_ANNOTATION_NOISE_ORI_STD_DEG="0"
EVAL_ANNOTATION_NOISE_SEED="0"
EVAL_ANNOTATION_NOISE_MODE="gaussian_clip_2sigma"
EVAL_ANNOTATION_NOISE_APPLY_TO="all"  # point, grasp, all

# Optional CLI override. If empty, it is derived from the local checkpoint filename (without extension).
ROLLOUT_SUFFIX_MODEL_NAME=""
OVERWRITE_WT_PATH=""  # 如果设置，直接使用此路径作为 --wt-path，跳过自动拼接和下载步骤

PARAMS=()

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

validate_annotation_flags() {
    if [[ "$EVAL_GRASP_PART_ANNOTATE" == "true" && "$EVAL_ANNOTATE_SKILL" != "true" ]]; then
        die "EVAL_GRASP_PART_ANNOTATE=true requires EVAL_ANNOTATE_SKILL=true"
    fi
    if [[ "$EVAL_GRASP_PART_ANNOTATE" == "true" && "$EVAL_GUIDANCE_POINT_ON_IMAGE" == "true" ]]; then
        die "EVAL_GRASP_PART_ANNOTATE=true cannot be combined with EVAL_GUIDANCE_POINT_ON_IMAGE=true"
    fi
    if [[ "$EVAL_GRASP_PART_ANNOTATE" == "true" && "$EVAL_GRASP_ANNOTATION_ON_IMAGE" == "true" ]]; then
        die "EVAL_GRASP_PART_ANNOTATE=true cannot be combined with EVAL_GRASP_ANNOTATION_ON_IMAGE=true"
    fi
    if [[ "$EVAL_GUIDANCE_POINT_COLORED" == "true" && "$EVAL_GUIDANCE_POINT_ON_IMAGE" != "true" && "$EVAL_GRASP_PART_ANNOTATE" != "true" ]]; then
        die "EVAL_GUIDANCE_POINT_COLORED=true requires EVAL_GUIDANCE_POINT_ON_IMAGE=true or EVAL_GRASP_PART_ANNOTATE=true"
    fi
    if [[ "$EVAL_GRASP_ANNOTATION_COLORED" == "true" && "$EVAL_GRASP_ANNOTATION_ON_IMAGE" != "true" && "$EVAL_GRASP_PART_ANNOTATE" != "true" ]]; then
        die "EVAL_GRASP_ANNOTATION_COLORED=true requires EVAL_GRASP_ANNOTATION_ON_IMAGE=true or EVAL_GRASP_PART_ANNOTATE=true"
    fi
}

build_params() {
    PARAMS=(
        --if-exists append
        --max-rollout-steps 1000
        --action-type pos
        --observation-space image
        --randomness "$RANDOMNESS"
        --save-rollouts
        --save-failures
        --save-depth-image  # disabled for state-based eval
    )
    if [[ "$MAX_SAVED_ROLLOUTS" != "0" ]]; then
        PARAMS+=(--max-saved-rollouts "$MAX_SAVED_ROLLOUTS")
    fi
    if [[ "$EVAL_ANNOTATE_SKILL" == "true" ]]; then
        PARAMS+=(--annotate-skill)
    fi
    if [[ "$EVAL_GUIDANCE_POINT_ON_IMAGE" == "true" ]]; then
        PARAMS+=(--guidance-point-on-image)
    fi
    if [[ "$EVAL_GRASP_ANNOTATION_ON_IMAGE" == "true" ]]; then
        PARAMS+=(--grasp-annotation-on-image)
    fi
    if [[ "$EVAL_GRASP_PART_ANNOTATE" == "true" ]]; then
        PARAMS+=(--grasp-part-annotate)
    fi
    if [[ "$EVAL_GUIDANCE_POINT_COLORED" == "true" ]]; then
        PARAMS+=(--guidance-point-colored)
    fi
    if [[ "$EVAL_GRASP_ANNOTATION_COLORED" == "true" ]]; then
        PARAMS+=(--grasp-annotation-colored)
    fi
    if [[ "$EVAL_ANNOTATE_SKILL" == "true" && "$EVAL_SKILL_ON_IMAGE" == "true" ]]; then
        PARAMS+=(--skill-on-image)
    fi
    if [[ "$EVAL_PERTURB_MODE" != "none" ]]; then
        PARAMS+=(--perturb-mode "$EVAL_PERTURB_MODE")
    fi
    if [[ "$EVAL_ANNOTATION_NOISE_POS_STD_M" != "0" || "$EVAL_ANNOTATION_NOISE_ORI_STD_DEG" != "0" ]]; then
        PARAMS+=(
            --annotation-noise-pos-std-m "$EVAL_ANNOTATION_NOISE_POS_STD_M"
            --annotation-noise-ori-std-deg "$EVAL_ANNOTATION_NOISE_ORI_STD_DEG"
            --annotation-noise-seed "$EVAL_ANNOTATION_NOISE_SEED"
            --annotation-noise-mode "$EVAL_ANNOTATION_NOISE_MODE"
            --annotation-noise-apply-to "$EVAL_ANNOTATION_NOISE_APPLY_TO"
        )
    fi
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
    elif [[ "$host" =~ ^zju_ ]]; then
        printf '%s\n' "$host"
    elif [[ "$host" =~ ^[0-9]+$ ]]; then
        local found
        found=$(awk '
            BEGIN { IGNORECASE = 1 }
            /^[[:space:]]*Host[[:space:]]+/ {
                for (i = 2; i <= NF; i++) {
                    if ($i ~ /^zju_/ && $i ~ /'"$host"'$/ && $i !~ /[*?]/) {
                        print $i; exit
                    }
                }
            }
        ' "$SSH_CONFIG_PATH" 2>/dev/null)
        if [[ -n "$found" ]]; then
            printf '%s\n' "$found"
        else
            printf 'zju_4090_%s\n' "$host"
        fi
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
import re
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


def has_any_epoch_suffix(checkpoint: Path) -> bool:
    return re.search(r"_\d+$", checkpoint.stem) is not None


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
        elif has_any_epoch_suffix(checkpoint):
            continue
        checkpoint_candidates.append((checkpoint.stat().st_mtime, str(checkpoint)))

    if not checkpoint_candidates:
        return None

    _mtime, checkpoint_text = max(checkpoint_candidates, key=lambda item: (item[0], item[1]))
    return checkpoint_text


selected_output_dir_text = None
selected_checkpoint_text = None

for _timestamp, output_dir_text in reversed(output_dir_candidates):
    checkpoint_text = find_checkpoint_in_output(Path(output_dir_text))
    if checkpoint_text is None:
        continue
    selected_output_dir_text = output_dir_text
    selected_checkpoint_text = checkpoint_text
    break

if selected_checkpoint_text is None:
    if epoch:
        raise SystemExit(
            f"No checkpoints matching pattern '{checkpoint_pattern}' with required suffix "
            f"'_{epoch}' found under outputs for RUN_ID '{run_id}'"
        )
    raise SystemExit(
        f"No checkpoints matching pattern '{checkpoint_pattern}' found under outputs for "
        f"RUN_ID '{run_id}'"
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
import re
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


def has_any_epoch_suffix(checkpoint: Path) -> bool:
    return re.search(r"_\d+$", checkpoint.stem) is not None


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
        elif has_any_epoch_suffix(checkpoint):
            continue
        checkpoint_candidates.append((checkpoint.stat().st_mtime, str(checkpoint)))

    if not checkpoint_candidates:
        return None

    _mtime, checkpoint_text = max(checkpoint_candidates, key=lambda item: (item[0], item[1]))
    return checkpoint_text


selected_output_dir_text = None
selected_checkpoint_text = None

for _timestamp, output_dir_text in reversed(output_dir_candidates):
    checkpoint_text = find_checkpoint_in_output(Path(output_dir_text))
    if checkpoint_text is None:
        continue
    selected_output_dir_text = output_dir_text
    selected_checkpoint_text = checkpoint_text
    break

if selected_checkpoint_text is None:
    if epoch:
        raise SystemExit(
            f"No checkpoints matching pattern '{checkpoint_pattern}' with required suffix "
            f"'_{epoch}' found under outputs for RUN_ID '{run_id}'"
        )
    raise SystemExit(
        f"No checkpoints matching pattern '{checkpoint_pattern}' found under outputs for "
        f"RUN_ID '{run_id}'"
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
        log_info "Searching matching output directories under ${outputs_root} for checkpoint without epoch suffix"
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
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hy/anaconda3/envs/rr/lib
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        export DATA_DIR_RAW="$DATA_DIR_RAW"
        "${eval_cmd[@]}"
    )
    log_info "Evaluation finished successfully."
}

print_usage() {
    cat <<'USAGE'
Usage: auto_eval.sh [options]

Options:
  --rollout-suffix-model-name NAME  Value passed to evaluator; defaults to local checkpoint filename stem.
  --steps "download eval"           Override STEPS for this invocation.
  --run-id NAME                     Override RUN_ID for this invocation.
  --project NAME                    Override PROJECT for this invocation.
  --task NAME                       Override TASK, e.g. one_leg+round_table+lamp.
  --remote-ssh-host HOST            Override REMOTE_SSH_HOST.
  --checkpoint-pattern PATTERN      Override CHECKPOINT_PATTERN.
  --epoch EPOCH                     Override EPOCH suffix filter.
  --n-envs N                        Override N_ENVS.
  --n-rollouts N                    Override N_ROLLOUTS.
  --randomness low|med|high         Override eval randomness.
  --max-saved-rollouts N            Save at most N rollout trajectories per task.
  --gpu-id ID                       Override GPU_ID.
  --overwrite-wt-path PATH          Override local checkpoint path and skip download.
  --noise-pos-std-m VALUE           Override annotation noise position std in meters.
  --noise-ori-std-deg VALUE         Override annotation noise orientation std in degrees.
  --noise-seed SEED                 Override annotation noise seed.
  --noise-mode MODE                 Override annotation noise mode.
  --noise-apply-to point|grasp|all  Override annotation noise target.
  --annotate-skill                  Enable skill annotation.
  --no-annotate-skill               Disable skill annotation.
  --guidance-point-on-image         Enable guidance point overlay.
  --grasp-annotation-on-image       Enable grasp annotation overlay.
  --grasp-part-annotate             Enable grasp-part annotation mode.
  --skill-on-image                  Enable skill label overlay.
  --guidance-point-colored          Enable colored guidance point annotation.
  --grasp-annotation-colored        Enable colored grasp annotation.
  --visualize                       Enable evaluator visualization.
  --debug                           Enable evaluator debug mode.
  -h, --help                        Show this help.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --steps)
                shift
                [[ $# -gt 0 ]] || die "--steps requires a value"
                read -r -a STEPS <<< "$1"
                shift
                ;;
            --run-id)
                shift
                [[ $# -gt 0 ]] || die "--run-id requires a value"
                RUN_ID="$1"
                shift
                ;;
            --project)
                shift
                [[ $# -gt 0 ]] || die "--project requires a value"
                PROJECT="$1"
                shift
                ;;
            --task)
                shift
                [[ $# -gt 0 ]] || die "--task requires a value"
                TASK="$1"
                shift
                ;;
            --remote-ssh-host)
                shift
                [[ $# -gt 0 ]] || die "--remote-ssh-host requires a value"
                REMOTE_SSH_HOST="$1"
                shift
                ;;
            --checkpoint-pattern)
                shift
                [[ $# -gt 0 ]] || die "--checkpoint-pattern requires a value"
                CHECKPOINT_PATTERN="$1"
                shift
                ;;
            --epoch)
                shift
                [[ $# -gt 0 ]] || die "--epoch requires a value"
                EPOCH="$1"
                shift
                ;;
            --n-envs)
                shift
                [[ $# -gt 0 ]] || die "--n-envs requires a value"
                N_ENVS="$1"
                shift
                ;;
            --n-rollouts)
                shift
                [[ $# -gt 0 ]] || die "--n-rollouts requires a value"
                N_ROLLOUTS="$1"
                shift
                ;;
            --randomness)
                shift
                [[ $# -gt 0 ]] || die "--randomness requires a value"
                RANDOMNESS="$1"
                shift
                ;;
            --max-saved-rollouts)
                shift
                [[ $# -gt 0 ]] || die "--max-saved-rollouts requires a value"
                MAX_SAVED_ROLLOUTS="$1"
                shift
                ;;
            --gpu-id)
                shift
                [[ $# -gt 0 ]] || die "--gpu-id requires a value"
                GPU_ID="$1"
                shift
                ;;
            --overwrite-wt-path)
                shift
                [[ $# -gt 0 ]] || die "--overwrite-wt-path requires a value"
                OVERWRITE_WT_PATH="$1"
                shift
                ;;
            --rollout-suffix-model-name)
                shift
                [[ $# -gt 0 ]] || die "--rollout-suffix-model-name requires a value"
                ROLLOUT_SUFFIX_MODEL_NAME="$1"
                shift
                ;;
            --noise-pos-std-m)
                shift
                [[ $# -gt 0 ]] || die "--noise-pos-std-m requires a value"
                EVAL_ANNOTATION_NOISE_POS_STD_M="$1"
                shift
                ;;
            --noise-ori-std-deg)
                shift
                [[ $# -gt 0 ]] || die "--noise-ori-std-deg requires a value"
                EVAL_ANNOTATION_NOISE_ORI_STD_DEG="$1"
                shift
                ;;
            --noise-seed)
                shift
                [[ $# -gt 0 ]] || die "--noise-seed requires a value"
                EVAL_ANNOTATION_NOISE_SEED="$1"
                shift
                ;;
            --noise-mode)
                shift
                [[ $# -gt 0 ]] || die "--noise-mode requires a value"
                EVAL_ANNOTATION_NOISE_MODE="$1"
                shift
                ;;
            --noise-apply-to)
                shift
                [[ $# -gt 0 ]] || die "--noise-apply-to requires a value"
                EVAL_ANNOTATION_NOISE_APPLY_TO="$1"
                shift
                ;;
            --annotate-skill)
                EVAL_ANNOTATE_SKILL=true
                shift
                ;;
            --no-annotate-skill)
                EVAL_ANNOTATE_SKILL=false
                shift
                ;;
            --guidance-point-on-image)
                EVAL_GUIDANCE_POINT_ON_IMAGE=true
                shift
                ;;
            --grasp-annotation-on-image)
                EVAL_GRASP_ANNOTATION_ON_IMAGE=true
                shift
                ;;
            --grasp-part-annotate)
                EVAL_GRASP_PART_ANNOTATE=true
                shift
                ;;
            --skill-on-image)
                EVAL_SKILL_ON_IMAGE=true
                shift
                ;;
            --guidance-point-colored)
                EVAL_GUIDANCE_POINT_COLORED=true
                shift
                ;;
            --grasp-annotation-colored)
                EVAL_GRASP_ANNOTATION_COLORED=true
                shift
                ;;
            --visualize)
                VISUALIZE=true
                shift
                ;;
            --debug)
                DEBUG=true
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
    validate_annotation_flags
    build_params

    local_root="$(expand_path "$LOCAL_PATH")"
    [[ -d "$local_root" ]] || die "LOCAL_PATH does not exist: ${local_root}"

    if [[ -n "${OVERWRITE_WT_PATH// }" ]]; then
        local_checkpoint="$OVERWRITE_WT_PATH"
    else
        checkpoint_name_tag="$(checkpoint_pattern_to_name_tag "$CHECKPOINT_PATTERN")"
        destination_dir="${local_root}/checkpoints/bc/${TASK}/low"
        local_checkpoint="${destination_dir}/${PROJECT}_${RUN_ID}_${checkpoint_name_tag}_${EPOCH}.pt"
    fi

    rollout_suffix_model_name="$ROLLOUT_SUFFIX_MODEL_NAME"
    if [[ -z "$rollout_suffix_model_name" ]]; then
        rollout_suffix_model_name="$(basename -- "$local_checkpoint")"
        rollout_suffix_model_name="${rollout_suffix_model_name%.*}"
    fi
    if [[ "$EVAL_ANNOTATION_NOISE_POS_STD_M" != "0" || "$EVAL_ANNOTATION_NOISE_ORI_STD_DEG" != "0" ]]; then
        local pos_tag ori_tag
        pos_tag="${EVAL_ANNOTATION_NOISE_POS_STD_M//./p}"
        ori_tag="${EVAL_ANNOTATION_NOISE_ORI_STD_DEG//./p}"
        rollout_suffix_model_name="${rollout_suffix_model_name}_noise_pos${pos_tag}_ori${ori_tag}_seed${EVAL_ANNOTATION_NOISE_SEED}"
    fi

    if [[ ${#STEPS[@]} -eq 0 ]]; then
        log_info "No steps selected in STEPS; nothing to do."
        return 0
    fi

    for step in "${STEPS[@]}"; do
        case "$step" in
            download)
                if [[ -n "${OVERWRITE_WT_PATH// }" ]]; then
                    log_info "OVERWRITE_WT_PATH is set, skipping download step."
                    continue
                fi
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
