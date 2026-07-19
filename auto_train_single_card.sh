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

annotation_die() {
    printf '[%s] ERROR %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
    exit 1
}

validate_dataset_annotation_config() {
    if [[ "$DATA_ANNOTATE_GRASP" == "true" && "$DATA_ANNOTATE_GRASP_PART" == "true" ]]; then
        annotation_die "DATA_ANNOTATE_GRASP and DATA_ANNOTATE_GRASP_PART cannot both be true"
    fi
    if [[ "$DATA_ANNOTATE_GRASP_COLORED" == "true" && "$DATA_ANNOTATE_GRASP" != "true" && "$DATA_ANNOTATE_GRASP_PART" != "true" ]]; then
        annotation_die "DATA_ANNOTATE_GRASP_COLORED=true requires DATA_ANNOTATE_GRASP=true or DATA_ANNOTATE_GRASP_PART=true"
    fi
    if [[ "$DATA_GUIDANCE_POINT_COLORED" == "true" && "$DATA_ANNOTATE_GUIDANCE_POINT" != "true" && "$DATA_ANNOTATE_GRASP_PART" != "true" ]]; then
        annotation_die "DATA_GUIDANCE_POINT_COLORED=true requires DATA_ANNOTATE_GUIDANCE_POINT=true or DATA_ANNOTATE_GRASP_PART=true"
    fi
    if [[ "$DATA_ANNOTATE_GRASP_PART" == "true" && "$DATA_GUIDANCE_POINT_COLORED" != "$DATA_ANNOTATE_GRASP_COLORED" ]]; then
        annotation_die "DATA_ANNOTATE_GRASP_PART=true requires DATA_GUIDANCE_POINT_COLORED and DATA_ANNOTATE_GRASP_COLORED to match"
    fi
}

build_data_suffix() {
    if [[ "$DATA_ANNOTATE_GRASP_PART" == "true" ]]; then
        if [[ "$DATA_ANNOTATE_GRASP_COLORED" == "true" ]]; then
            printf 'rgbd-skill-grasp-part-colored\n'
        else
            printf 'rgbd-skill-grasp-part\n'
        fi
        return
    fi
    if [[ "$DATA_ANNOTATE_GRASP" == "true" ]]; then
        if [[ "$DATA_ANNOTATE_GRASP_COLORED" == "true" ]]; then
            printf 'rgbd-skill-grasp-colored\n'
        else
            printf 'rgbd-skill-grasp\n'
        fi
        return
    fi
    if [[ "$DATA_ANNOTATE_GUIDANCE_POINT" == "true" ]]; then
        if [[ "$DATA_GUIDANCE_POINT_COLORED" == "true" ]]; then
            printf 'rgbd-skill-point-colored\n'
        else
            printf 'rgbd-skill-point\n'
        fi
        return
    fi
    if [[ "$DATA_ANNOTATE_SKILL_ONE_HOT" == "true" ]]; then
        printf 'rgbd-only-skill\n'
        return
    fi
    printf 'rgbd\n'
}

format_noise_component() {
    python3 - "$1" "$2" <<'PY'
import math
import re
import sys

prefix = sys.argv[1]
value = float(sys.argv[2])
if not math.isfinite(value):
    raise SystemExit(f"{prefix} noise value must be finite, got {value!r}")
text = f"{value:g}".replace("-", "m").replace(".", "p")
text = re.sub(r"[^A-Za-z0-9]+", "_", text)
print(f"{prefix}{text}")
PY
}

sanitize_suffix_component() {
    python3 - "$1" <<'PY'
import re
import sys

text = sys.argv[1].strip()
text = re.sub(r"[^A-Za-z0-9]+", "_", text)
print(text.strip("_") or "unknown")
PY
}

build_noise_suffix() {
    local pos_std_m="$1"
    local ori_std_deg="$2"
    local mode="$3"
    local apply_to="$4"
    local seed="$5"
    local pos_mm
    local nonzero

    nonzero="$(python3 - "$pos_std_m" "$ori_std_deg" <<'PY'
import sys
print("1" if float(sys.argv[1]) != 0.0 or float(sys.argv[2]) != 0.0 else "0")
PY
)"
    if [[ "$nonzero" == "0" ]]; then
        printf '%s\n' ""
        return
    fi

    pos_mm="$(python3 - "$pos_std_m" <<'PY'
import sys
print(float(sys.argv[1]) * 1000.0)
PY
)"
    printf 'noise-%s-%s-mode%s-apply%s-seed%s\n' \
        "$(format_noise_component pos "$pos_mm")" \
        "$(format_noise_component ori "$ori_std_deg")" \
        "$(sanitize_suffix_component "$mode")" \
        "$(sanitize_suffix_component "$apply_to")" \
        "$seed"
}

print_usage() {
    cat <<'USAGE'
Usage: auto_train_single_card.sh [options]

Manual defaults live near the top of this script. CLI options override those defaults
and any *_OVERRIDE environment variables for one run.

Dataset/model:
  --experiment NAME
  --task-spec "[one_leg,round_table,lamp]"
  --data-suffix SUFFIX
  --data-suffix-fallback SUFFIX
  --data-paths-override VALUE
  --storage-format FORMAT
  --load-into-memory true|false
  --guidance-point / --no-guidance-point
  --skill-one-hot / --no-skill-one-hot
  --guidance-point-colored / --no-guidance-point-colored
  --grasp / --no-grasp
  --grasp-colored / --no-grasp-colored
  --grasp-part / --no-grasp-part
  --noise-pos-std-m M
  --noise-ori-std-deg DEG
  --noise-seed SEED
  --noise-mode MODE
  --noise-apply-to point|grasp|all

Training/runtime:
  --batch-size N
  --num-epochs N
  --steps-per-epoch N
  --save-per-epoch N
  --dataloader-workers N
  --data-subset N
  --randomness low|med|high
  --dryrun true|false
  --wandb-project NAME
  --wandb-mode online|offline|disabled
  --vision-encoder-pretrained true|false
  --gpu-id ID
  --ssh-name HOST
  --data-dir-processed DIR
  --remote-project-dir DIR
  --remote-conda-env ENV
  --checkpoint-fallback-dir DIR
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                print_usage
                exit 0
                ;;
            --experiment)
                EXPERIMENT_NAME="${2:?Missing value for $1}"
                shift 2
                ;;
            --task-spec)
                TASK_SPEC="${2:?Missing value for $1}"
                shift 2
                ;;
            --data-suffix)
                DATA_SUFFIX_OVERRIDE="${2:?Missing value for $1}"
                shift 2
                ;;
            --data-suffix-fallback)
                DATA_SUFFIX_FALLBACK="${2:?Missing value for $1}"
                shift 2
                ;;
            --data-paths-override)
                DATA_PATHS_OVERRIDE="${2:?Missing value for $1}"
                shift 2
                ;;
            --storage-format)
                DATA_STORAGE_FORMAT="${2:?Missing value for $1}"
                shift 2
                ;;
            --load-into-memory)
                DATA_LOAD_INTO_MEMORY="${2:?Missing value for $1}"
                shift 2
                ;;
            --guidance-point)
                DATA_ANNOTATE_GUIDANCE_POINT=true
                shift
                ;;
            --no-guidance-point)
                DATA_ANNOTATE_GUIDANCE_POINT=false
                shift
                ;;
            --skill-one-hot)
                DATA_ANNOTATE_SKILL_ONE_HOT=true
                shift
                ;;
            --no-skill-one-hot)
                DATA_ANNOTATE_SKILL_ONE_HOT=false
                shift
                ;;
            --guidance-point-colored)
                DATA_GUIDANCE_POINT_COLORED=true
                shift
                ;;
            --no-guidance-point-colored)
                DATA_GUIDANCE_POINT_COLORED=false
                shift
                ;;
            --grasp)
                DATA_ANNOTATE_GRASP=true
                shift
                ;;
            --no-grasp)
                DATA_ANNOTATE_GRASP=false
                shift
                ;;
            --grasp-colored)
                DATA_ANNOTATE_GRASP_COLORED=true
                shift
                ;;
            --no-grasp-colored)
                DATA_ANNOTATE_GRASP_COLORED=false
                shift
                ;;
            --grasp-part)
                DATA_ANNOTATE_GRASP_PART=true
                shift
                ;;
            --no-grasp-part)
                DATA_ANNOTATE_GRASP_PART=false
                shift
                ;;
            --noise-pos-std-m)
                DATA_ANNOTATION_NOISE_POS_STD_M="${2:?Missing value for $1}"
                shift 2
                ;;
            --noise-ori-std-deg)
                DATA_ANNOTATION_NOISE_ORI_STD_DEG="${2:?Missing value for $1}"
                shift 2
                ;;
            --noise-seed)
                DATA_ANNOTATION_NOISE_SEED="${2:?Missing value for $1}"
                shift 2
                ;;
            --noise-mode)
                DATA_ANNOTATION_NOISE_MODE="${2:?Missing value for $1}"
                shift 2
                ;;
            --noise-apply-to)
                DATA_ANNOTATION_NOISE_APPLY_TO="${2:?Missing value for $1}"
                shift 2
                ;;
            --batch-size)
                TRAIN_BATCH_SIZE="${2:?Missing value for $1}"
                shift 2
                ;;
            --num-epochs)
                TRAIN_NUM_EPOCHS="${2:?Missing value for $1}"
                shift 2
                ;;
            --steps-per-epoch)
                TRAIN_STEPS_PER_EPOCH="${2:?Missing value for $1}"
                shift 2
                ;;
            --save-per-epoch)
                TRAIN_SAVE_PER_EPOCH="${2:?Missing value for $1}"
                shift 2
                ;;
            --dataloader-workers)
                DATA_DATALOADER_WORKERS="${2:?Missing value for $1}"
                shift 2
                ;;
            --data-subset)
                DATA_DATA_SUBSET="${2:?Missing value for $1}"
                shift 2
                ;;
            --randomness)
                TRAIN_RANDOMNESS="${2:?Missing value for $1}"
                shift 2
                ;;
            --dryrun)
                TRAIN_DRYRUN="${2:?Missing value for $1}"
                shift 2
                ;;
            --wandb-project)
                WANDB_PROJECT="${2:?Missing value for $1}"
                shift 2
                ;;
            --wandb-mode)
                WANDB_MODE="${2:?Missing value for $1}"
                shift 2
                ;;
            --vision-encoder-pretrained)
                VISION_ENCODER_PRETRAINED="${2:?Missing value for $1}"
                shift 2
                ;;
            --gpu-id)
                GPU_ID="${2:?Missing value for $1}"
                shift 2
                ;;
            --ssh-name)
                SSH_NAME="${2:?Missing value for $1}"
                shift 2
                ;;
            --data-dir-processed)
                DATA_DIR_PROCESSED="${2:?Missing value for $1}"
                shift 2
                ;;
            --remote-project-dir)
                REMOTE_PROJECT_DIR="${2:?Missing value for $1}"
                shift 2
                ;;
            --remote-conda-env)
                REMOTE_CONDA_ENV="${2:?Missing value for $1}"
                shift 2
                ;;
            --checkpoint-fallback-dir)
                CHECKPOINT_FALLBACK_DIR="${2:?Missing value for $1}"
                shift 2
                ;;
            *)
                printf 'Unknown argument: %s\n\n' "$1" >&2
                print_usage >&2
                exit 2
                ;;
        esac
    done
}

DATA_STORAGE_FORMAT="lmdb"
DATA_LOAD_INTO_MEMORY="false"
DATA_PATHS_OVERRIDE=""
CHECKPOINT_FALLBACK_DIR="/home/hy/tmp/checkpoint_fallback"
DATA_ANNOTATE_GUIDANCE_POINT="true"
DATA_ANNOTATE_SKILL_ONE_HOT="false"
DATA_GUIDANCE_POINT_COLORED="false"  # yellow=pick/screw, red=place/push/insert
DATA_ANNOTATE_GRASP="false"
DATA_ANNOTATE_GRASP_COLORED="false"
DATA_ANNOTATE_GRASP_PART="false"
DATA_ANNOTATION_NOISE_POS_STD_M="0"
DATA_ANNOTATION_NOISE_ORI_STD_DEG="0"
DATA_ANNOTATION_NOISE_SEED="0"
DATA_ANNOTATION_NOISE_MODE="gaussian_clip_2sigma"
DATA_ANNOTATION_NOISE_APPLY_TO="all"
DATA_SUFFIX_FALLBACK=""
EXPERIMENT_NAME="rgbd/diff_unet"
VISION_ENCODER_PRETRAINED="false"
TASK_SPEC="[one_leg,round_table,lamp]"
DATA_DATALOADER_WORKERS=4
DATA_DATA_SUBSET=50
TRAIN_BATCH_SIZE=256
TRAIN_NUM_EPOCHS=5000
TRAIN_STEPS_PER_EPOCH=-1
TRAIN_SAVE_PER_EPOCH=1000
TRAIN_RANDOMNESS="low"
TRAIN_DRYRUN="false"
WANDB_PROJECT="multi-task-rgbd-skill-low-smalldot"
WANDB_MODE="online"
TRAIN_GPU_ID=0
SSH_NAME="230"
GPU_ID="0"
# DATA_DIR_PROCESSED="/data/hy/robust-rearrangement-custom/data/"  # server local
DATA_DIR_PROCESSED="~/robust-rearrangement-custom/data/"  # home, for 236
RUNTIME_TMP_ROOT="${RUNTIME_TMP_ROOT:-/home/hy/tmp}"  # local tmp, NOT NAS
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/mnt/nas/share/home/hy/robust-rearrangement-custom}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-rr}"
DATA_STORAGE_FORMAT="${DATA_STORAGE_FORMAT_OVERRIDE:-$DATA_STORAGE_FORMAT}"
DATA_LOAD_INTO_MEMORY="${DATA_LOAD_INTO_MEMORY_OVERRIDE:-$DATA_LOAD_INTO_MEMORY}"
DATA_PATHS_OVERRIDE="${DATA_PATHS_OVERRIDE_OVERRIDE:-$DATA_PATHS_OVERRIDE}"
CHECKPOINT_FALLBACK_DIR="${CHECKPOINT_FALLBACK_DIR_OVERRIDE:-$CHECKPOINT_FALLBACK_DIR}"
DATA_ANNOTATE_GUIDANCE_POINT="${DATA_ANNOTATE_GUIDANCE_POINT_OVERRIDE:-$DATA_ANNOTATE_GUIDANCE_POINT}"
DATA_ANNOTATE_SKILL_ONE_HOT="${DATA_ANNOTATE_SKILL_ONE_HOT_OVERRIDE:-$DATA_ANNOTATE_SKILL_ONE_HOT}"
DATA_GUIDANCE_POINT_COLORED="${DATA_GUIDANCE_POINT_COLORED_OVERRIDE:-$DATA_GUIDANCE_POINT_COLORED}"
DATA_ANNOTATE_GRASP="${DATA_ANNOTATE_GRASP_OVERRIDE:-$DATA_ANNOTATE_GRASP}"
DATA_ANNOTATE_GRASP_COLORED="${DATA_ANNOTATE_GRASP_COLORED_OVERRIDE:-$DATA_ANNOTATE_GRASP_COLORED}"
DATA_ANNOTATE_GRASP_PART="${DATA_ANNOTATE_GRASP_PART_OVERRIDE:-$DATA_ANNOTATE_GRASP_PART}"
DATA_ANNOTATION_NOISE_POS_STD_M="${DATA_ANNOTATION_NOISE_POS_STD_M_OVERRIDE:-$DATA_ANNOTATION_NOISE_POS_STD_M}"
DATA_ANNOTATION_NOISE_ORI_STD_DEG="${DATA_ANNOTATION_NOISE_ORI_STD_DEG_OVERRIDE:-$DATA_ANNOTATION_NOISE_ORI_STD_DEG}"
DATA_ANNOTATION_NOISE_SEED="${DATA_ANNOTATION_NOISE_SEED_OVERRIDE:-$DATA_ANNOTATION_NOISE_SEED}"
DATA_ANNOTATION_NOISE_MODE="${DATA_ANNOTATION_NOISE_MODE_OVERRIDE:-$DATA_ANNOTATION_NOISE_MODE}"
DATA_ANNOTATION_NOISE_APPLY_TO="${DATA_ANNOTATION_NOISE_APPLY_TO_OVERRIDE:-$DATA_ANNOTATION_NOISE_APPLY_TO}"
DATA_SUFFIX_FALLBACK="${DATA_SUFFIX_FALLBACK_OVERRIDE:-$DATA_SUFFIX_FALLBACK}"
EXPERIMENT_NAME="${EXPERIMENT_NAME_OVERRIDE:-$EXPERIMENT_NAME}"
VISION_ENCODER_PRETRAINED="${VISION_ENCODER_PRETRAINED_OVERRIDE:-$VISION_ENCODER_PRETRAINED}"
TASK_SPEC="${TASK_SPEC_OVERRIDE:-$TASK_SPEC}"
DATA_DATALOADER_WORKERS="${DATA_DATALOADER_WORKERS_OVERRIDE:-$DATA_DATALOADER_WORKERS}"
DATA_DATA_SUBSET="${DATA_DATA_SUBSET_OVERRIDE:-$DATA_DATA_SUBSET}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_OVERRIDE:-$TRAIN_BATCH_SIZE}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS_OVERRIDE:-$TRAIN_NUM_EPOCHS}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH_OVERRIDE:-$TRAIN_STEPS_PER_EPOCH}"
TRAIN_SAVE_PER_EPOCH="${TRAIN_SAVE_PER_EPOCH_OVERRIDE:-$TRAIN_SAVE_PER_EPOCH}"
TRAIN_RANDOMNESS="${TRAIN_RANDOMNESS_OVERRIDE:-$TRAIN_RANDOMNESS}"
TRAIN_DRYRUN="${TRAIN_DRYRUN_OVERRIDE:-$TRAIN_DRYRUN}"
WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-$WANDB_PROJECT}"
WANDB_MODE="${WANDB_MODE_OVERRIDE:-$WANDB_MODE}"
TRAIN_GPU_ID="${TRAIN_GPU_ID_OVERRIDE:-$TRAIN_GPU_ID}"
SSH_NAME="${SSH_NAME_OVERRIDE:-$SSH_NAME}"
GPU_ID="${GPU_ID_OVERRIDE:-$GPU_ID}"
DATA_DIR_PROCESSED="${DATA_DIR_PROCESSED_OVERRIDE:-$DATA_DIR_PROCESSED}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR_OVERRIDE:-$REMOTE_PROJECT_DIR}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV_OVERRIDE:-$REMOTE_CONDA_ENV}"
parse_args "$@"
validate_dataset_annotation_config
DATA_SUFFIX="$(build_data_suffix)"
DATA_NOISE_SUFFIX="$(build_noise_suffix "$DATA_ANNOTATION_NOISE_POS_STD_M" "$DATA_ANNOTATION_NOISE_ORI_STD_DEG" "$DATA_ANNOTATION_NOISE_MODE" "$DATA_ANNOTATION_NOISE_APPLY_TO" "$DATA_ANNOTATION_NOISE_SEED")"
if [[ -n "$DATA_NOISE_SUFFIX" ]]; then
    DATA_SUFFIX="${DATA_SUFFIX}-${DATA_NOISE_SUFFIX}"
fi
DATA_SUFFIX="${DATA_SUFFIX_OVERRIDE:-$DATA_SUFFIX}"

# Single-card training command.
TRAIN_COMMAND_PARTS=(
    python
    -m
    src.train.bc
    "+experiment=${EXPERIMENT_NAME}"
    "vision_encoder.pretrained=${VISION_ENCODER_PRETRAINED}"
    "task=${TASK_SPEC}"
    data.demo_source=rollout
    data.demo_outcome=success
    "data.suffix=${DATA_SUFFIX}"
    "data.annotate_guidance_point=${DATA_ANNOTATE_GUIDANCE_POINT}"
    "data.annotate_guidance_point_colored=${DATA_GUIDANCE_POINT_COLORED}"
    "data.annotate_grasp=${DATA_ANNOTATE_GRASP}"
    "data.annotate_grasp_colored=${DATA_ANNOTATE_GRASP_COLORED}"
    "data.annotate_grasp_part=${DATA_ANNOTATE_GRASP_PART}"
    "data.annotate_skill_one_hot=${DATA_ANNOTATE_SKILL_ONE_HOT}"
    "data.annotation_noise_pos_std_m=${DATA_ANNOTATION_NOISE_POS_STD_M}"
    "data.annotation_noise_ori_std_deg=${DATA_ANNOTATION_NOISE_ORI_STD_DEG}"
    "data.annotation_noise_seed=${DATA_ANNOTATION_NOISE_SEED}"
    "data.annotation_noise_mode=${DATA_ANNOTATION_NOISE_MODE}"
    "data.annotation_noise_apply_to=${DATA_ANNOTATION_NOISE_APPLY_TO}"
    "data.suffix_fallback=${DATA_SUFFIX_FALLBACK}"
    "data.storage_format=${DATA_STORAGE_FORMAT}"
    "data.load_into_memory=${DATA_LOAD_INTO_MEMORY}"
    "data.dataloader_workers=${DATA_DATALOADER_WORKERS}"
    "data.data_subset=${DATA_DATA_SUBSET}"
    "training.batch_size=${TRAIN_BATCH_SIZE}"
    "training.num_epochs=${TRAIN_NUM_EPOCHS}"
    "training.steps_per_epoch=${TRAIN_STEPS_PER_EPOCH}"
    "training.save_per_epoch=${TRAIN_SAVE_PER_EPOCH}"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.mode=${WANDB_MODE}"
    "training.gpu_id=${TRAIN_GPU_ID}"
    "randomness=${TRAIN_RANDOMNESS}"
    "dryrun=${TRAIN_DRYRUN}"
)

# Optional explicit dataset override.
if [[ -n "${DATA_PATHS_OVERRIDE// }" ]]; then
    TRAIN_COMMAND_PARTS+=("data.data_paths_override=${DATA_PATHS_OVERRIDE}")
fi
if [[ -n "${CHECKPOINT_FALLBACK_DIR// }" ]]; then
    TRAIN_COMMAND_PARTS+=("training.checkpoint_fallback_dir=${CHECKPOINT_FALLBACK_DIR}")
fi
TRAIN_COMMAND="$(join_command_parts "${TRAIN_COMMAND_PARTS[@]}")"
WANDB_PROJECT_NAME="$(get_command_part_value wandb.project "${TRAIN_COMMAND_PARTS[@]}" || printf 'project')"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-project}"
FAST_SERVER=(236 230)
SLOW_SERVER=(228 238 240 221 251 181 183)

SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
MEMORY_USAGE_THRESHOLD="${MEMORY_USAGE_THRESHOLD:-0.1}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-5}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-5}"
POLL_TIMEOUT_SECONDS="${POLL_TIMEOUT_SECONDS:-300}"
SSH_COMMAND_TIMEOUT_SECONDS="${SSH_COMMAND_TIMEOUT_SECONDS:-15}"
SSH_COMMON_ARGS=(
    -o BatchMode=yes
    -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=1
)

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
                if ($i ~ /^zju_/ && $i !~ /[*?]/) {
                    print $i
                }
            }
        }
    ' "$SSH_CONFIG_PATH" | awk '!seen[$0]++'
}

normalize_ssh_host() {
    local host="${1:-}"
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

run_ssh() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "${SSH_COMMAND_TIMEOUT_SECONDS}s" \
            ssh \
            "${SSH_COMMON_ARGS[@]}" \
            "$@"
    else
        ssh \
            "${SSH_COMMON_ARGS[@]}" \
            "$@"
    fi
}

capture_ssh_output() {
    local output_var="$1"
    shift
    local output
    local ssh_status
    local restore_errexit=0

    case $- in
        *e*)
            restore_errexit=1
            set +e
            ;;
    esac

    output="$(run_ssh "$@" 2>&1)"
    ssh_status=$?

    if (( restore_errexit )); then
        set -e
    fi

    printf -v "$output_var" '%s' "$output"
    return "$ssh_status"
}

encode_transport_field() {
    printf '%s' "$1" | base64 | tr -d '\r\n'
}

decode_transport_field() {
    python3 - "$1" <<'PY'
import base64
import sys

text = sys.argv[1]
if not text.strip():
    print("", end="")
    raise SystemExit(0)

try:
    decoded = base64.b64decode(text).decode("utf-8", "replace")
except Exception:
    decoded = text

print(decoded, end="")
PY
}

invoke_ssh() {
    local host_alias="$1"
    local remote_command="$2"

    run_ssh "$host_alias" "$remote_command"
}

get_host_gpu_status() {
    local host_alias="$1"
    local query_output

    if capture_ssh_output query_output \
        "$host_alias" \
        "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits"; then
        :
    else
        local ssh_status="$?"
        if [[ -z "$query_output" ]]; then
            query_output="SSH command failed with exit code ${ssh_status}."
        fi
        printf 'HOST|%s|DOWN|%s\n' "$host_alias" "$(encode_transport_field "$query_output")"
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
    local host_alias
    local selected
    local host_state="DOWN"
    local host_note=""
    local line
    local -a seen_gpu_ids=()

    if [[ ! "$gpu_id_text" =~ ^[0-9]+$ ]]; then
        echo "GPU_ID must be a non-negative integer, got '$gpu_id_text'." >&2
        return 1
    fi

    host_alias="$(normalize_ssh_host "$ssh_name")"

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
        local decoded_note
        decoded_note="$(decode_transport_field "$host_note")"
        if [[ -n "$decoded_note" ]]; then
            printf "Preferred host '%s' is unreachable: %s\n" "$host_alias" "$decoded_note" >&2
        else
            printf "Preferred host '%s' is unreachable.\n" "$host_alias" >&2
        fi
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

    run_ssh \
        "$host_alias" \
        bash -s -- "$session_name" "$REMOTE_PROJECT_DIR" "$REMOTE_CONDA_ENV" "$encoded_command" "$DATA_DIR_PROCESSED" "$WANDB_PROJECT_NAME" "$RUNTIME_TMP_ROOT" <<'REMOTE'
set -euo pipefail

session_name="$1"
project_dir="$2"
conda_env="$3"
encoded_train_command="$4"
data_dir_processed="${5:-}"
wandb_project_name="${6:-project}"
runtime_tmp_root="${7:-/data/hy/tmp}"
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
runtime_tmp_dir="${runtime_tmp_root}/wandb-${wandb_project_slug}"
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
    python3 /dev/fd/3 3<<'PY'
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
    python3 /dev/fd/3 3<<'PY'
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
