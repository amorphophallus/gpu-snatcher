#!/usr/bin/env bash

set -euo pipefail

# User-editable configuration

# Comment out a line to skip that step.
STEPS=(
    # collect_data
    # process_pickles
    upload
)

LOCAL_PATH="/data/hy/robust-rearrangement"  # 218
# LOCAL_PATH="~/projects/robust-rearrangement-custom"  # base
REMOTE_PATH="/data/hy/robust-rearrangement-custom/"  # server local
# REMOTE_PATH="~/robust-rearrangement-custom/"  # server local home, for 236
# REMOTE_PATH="/mnt/nas/share/home/hy/robust-rearrangement-custom/"  # NAS
REMOTE_SSH_HOST="228"
CONDA_ENV="rr"
CONNECT_TIMEOUT_SECONDS=10
UPLOAD_MAX_RETRIES=5
UPLOAD_RETRY_DELAY_SECONDS=5
SSH_STRICT_HOST_KEY_CHECKING="${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
SSH_SERVER_ALIVE_INTERVAL_SECONDS="${SSH_SERVER_ALIVE_INTERVAL_SECONDS:-15}"
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-12}"
UPLOAD_BWLIMIT="${UPLOAD_BWLIMIT:-100m}"

# Comment out a line to skip that task for collect/process.
TASKS=(
    # one_leg
    round_table
    # lamp
)

declare -A TASK_CKPT=(  # relative to local root
    [one_leg]="/checkpoints/rppo/one_leg/low/actor_chkpt.pt"
    [round_table]="/checkpoints/rppo/round_table/low/actor_chkpt.pt"
    [lamp]="/checkpoints/rppo/lamp/low/actor_chkpt.pt"
)

declare -A TASK_MAX_ROLLOUT_STEPS=(
    [one_leg]=700
    [round_table]=1000
    [lamp]=1000
)

declare -A TASK_ROLLOUT_AFTER_SUCCESS=(
    [one_leg]=200
    [round_table]=50
    [lamp]=20
)

COLLECT_N_ENVS=4
COLLECT_N_ROLLOUTS=200  # 要多少数据
COLLECT_IF_EXISTS="append"
COLLECT_ACTION_TYPE="pos"
COLLECT_OBSERVATION_SPACE="image"
COLLECT_RANDOMNESS="low"

declare -A TASK_EPISODE_LIMIT=(
    [one_leg]="$COLLECT_N_ROLLOUTS"
    [round_table]="$COLLECT_N_ROLLOUTS"
    [lamp]="$COLLECT_N_ROLLOUTS"
)

# # 自定义数据配比
# declare -A TASK_EPISODE_LIMIT=(
#     [one_leg]=1
#     [round_table]=1
#     [lamp]=1
# )

COLLECT_FLAGS=(
    --save-rollouts
    --save-depth-image
    --annotate-skill
    --skill-on-image
    --output-only-pickle
)

PROCESS_CONTROLLER="diffik"
PROCESS_DOMAIN="sim"
PROCESS_SOURCE="rollout"
PROCESS_RANDOMNESS="low"
PROCESS_OUTCOME="success"
PROCESS_SUFFIX="rgbd-skill"
PROCESS_OUTPUT_SUFFIX="rgbd-skill"
PROCESS_BATCH_SIZE=2
PYTHON_RUNTIME_CACHE_ROOT="${PYTHON_RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/gpu-snatcher-auto-data-preparation}"

PROCESS_FLAGS=(
    --overwrite
)

UPLOAD_RELATIVE_DIR="data/processed/diffik/sim"

# Functions

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

resolve_command_path() {
    local cmd="$1"

    if command -v "$cmd" >/dev/null 2>&1; then
        command -v "$cmd"
    elif [[ -x "/usr/bin/${cmd}" ]]; then
        printf '%s\n' "/usr/bin/${cmd}"
    elif [[ -x "/bin/${cmd}" ]]; then
        printf '%s\n' "/bin/${cmd}"
    else
        die "Required command not found: $cmd"
    fi
}

run_with_retry() {
    local max_retries="$1"
    local retry_delay_seconds="$2"
    local description="$3"
    shift 3

    local attempt exit_code

    attempt=1
    while (( attempt <= max_retries )); do
        log_info "${description} (attempt ${attempt}/${max_retries})"
        if "$@"; then
            if (( attempt > 1 )); then
                log_info "${description} succeeded on attempt ${attempt}/${max_retries}"
            fi
            return 0
        else
            exit_code=$?
        fi
        if (( attempt == max_retries )); then
            log_error "${description} failed on attempt ${attempt}/${max_retries} with exit code ${exit_code}"
            return "$exit_code"
        fi

        log_info "${description} failed on attempt ${attempt}/${max_retries} with exit code ${exit_code}; retrying in ${retry_delay_seconds}s"
        sleep "$retry_delay_seconds"
        ((attempt++))
    done
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

require_tasks_configured() {
    [[ ${#TASKS[@]} -gt 0 ]] || die "TASKS must contain at least one task."
}

get_sorted_tasks() {
    require_tasks_configured
    printf '%s\n' "${TASKS[@]}" | LC_ALL=C sort
}

get_task_group_name() {
    local -a sorted_tasks
    mapfile -t sorted_tasks < <(get_sorted_tasks)
    local IFS='-'
    printf '%s\n' "${sorted_tasks[*]}"
}

build_task_episode_limit_args() {
    local task episode_limit

    require_tasks_configured
    for task in "${TASKS[@]}"; do
        [[ -n "${TASK_EPISODE_LIMIT[$task]+x}" ]] || die "TASK_EPISODE_LIMIT is missing task: ${task}"
        episode_limit="${TASK_EPISODE_LIMIT[$task]}"
        [[ "$episode_limit" =~ ^[0-9]+$ ]] || die "TASK_EPISODE_LIMIT[$task] must be a non-negative integer, got: ${episode_limit}"
        printf '%s\n' "${task}=${episode_limit}"
    done
}

get_processed_dataset_relative_path() {
    local task_group dataset_path

    task_group="$(get_task_group_name)"
    dataset_path="${UPLOAD_RELATIVE_DIR%/}/${task_group}/${PROCESS_SOURCE}/${PROCESS_RANDOMNESS}/${PROCESS_OUTCOME}"

    if [[ -n "$PROCESS_OUTPUT_SUFFIX" ]]; then
        dataset_path="${dataset_path}/${PROCESS_OUTPUT_SUFFIX}.lmdb"
    else
        dataset_path="${dataset_path}.lmdb"
    fi

    printf '%s\n' "$dataset_path"
}

get_processed_dataset_absolute_path() {
    local local_root="$1"
    printf '%s/%s\n' "${local_root%/}" "$(get_processed_dataset_relative_path)"
}

get_conda_executable() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        printf '%s\n' "$HOME/miniconda3/bin/conda"
    elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
        printf '%s\n' "$HOME/anaconda3/bin/conda"
    else
        die "Unable to locate the conda executable. Please install conda or update get_conda_executable()."
    fi
}

sanitize_ld_library_path() {
    local raw_value="${1:-}"
    local IFS=':'
    local part
    local -a parts=()
    local -a filtered_parts=()

    read -r -a parts <<< "$raw_value"
    for part in "${parts[@]}"; do
        [[ -n "$part" ]] || continue
        case "$part" in
            "$HOME"/anaconda3/lib|"$HOME"/anaconda3/envs/*/lib|"$HOME"/miniconda3/lib|"$HOME"/miniconda3/envs/*/lib)
                continue
                ;;
        esac
        filtered_parts+=("$part")
    done

    (
        local IFS=':'
        printf '%s\n' "${filtered_parts[*]}"
    )
}

ensure_python_runtime_dirs() {
    local cache_root="$1"

    mkdir -p "${cache_root%/}/matplotlib" "${cache_root%/}/python"
}

run_python_command() {
    local local_root="$1"
    shift
    local conda_exe sanitized_ld_library_path
    local -a cmd=( "$@" )

    conda_exe="$(get_conda_executable)"
    ensure_python_runtime_dirs "$PYTHON_RUNTIME_CACHE_ROOT"
    sanitized_ld_library_path="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"

    log_info "Running command in ${local_root} with conda env ${CONDA_ENV}: $(quote_command "${cmd[@]}")"
    env \
        LD_LIBRARY_PATH="$sanitized_ld_library_path" \
        MPLCONFIGDIR="${PYTHON_RUNTIME_CACHE_ROOT%/}/matplotlib" \
        PYTHONNOUSERSITE=1 \
        PYTHONPYCACHEPREFIX="${PYTHON_RUNTIME_CACHE_ROOT%/}/python" \
        "$conda_exe" \
        run \
        --cwd "$local_root" \
        --no-capture-output \
        -n "$CONDA_ENV" \
        "${cmd[@]}"
}

check_process_runtime_dependencies() {
    local local_root="$1"

    run_python_command "$local_root" python - <<'PY'
import importlib
import sys
import traceback

required_modules = ("lmdb",)
failures = []

for module_name in required_modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - shell preflight
        failures.append((module_name, exc, traceback.format_exc()))

if failures:
    print("Python dependency check failed before process_pickles_to_lmdb.", file=sys.stderr)
    for module_name, exc, tb in failures:
        print(f"[dependency] {module_name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
    print(
        "Reinstall the broken package inside the target env, for example: "
        "conda run -n rr python -m pip install --force-reinstall lmdb",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

collect_data_step() {
    local local_root="$1"
    local task checkpoint_path max_rollout_steps rollout_after_success
    local -a collect_cmd

    for task in "${TASKS[@]}"; do
        [[ -n "${TASK_CKPT[$task]+x}" ]] || die "TASK_CKPT is missing task: ${task}"
        [[ -n "${TASK_MAX_ROLLOUT_STEPS[$task]+x}" ]] || die "TASK_MAX_ROLLOUT_STEPS is missing task: ${task}"
        [[ -n "${TASK_ROLLOUT_AFTER_SUCCESS[$task]+x}" ]] || die "TASK_ROLLOUT_AFTER_SUCCESS is missing task: ${task}"

        checkpoint_path="${local_root%/}${TASK_CKPT[$task]}"
        max_rollout_steps="${TASK_MAX_ROLLOUT_STEPS[$task]}"
        rollout_after_success="${TASK_ROLLOUT_AFTER_SUCCESS[$task]}"

        collect_cmd=(
            python -m src.eval.evaluate_model
            --n-envs "$COLLECT_N_ENVS"
            --n-rollouts "$COLLECT_N_ROLLOUTS"
            -f "$task"
            --if-exists "$COLLECT_IF_EXISTS"
            --max-rollout-steps "$max_rollout_steps"
            --action-type "$COLLECT_ACTION_TYPE"
            --observation-space "$COLLECT_OBSERVATION_SPACE"
            --randomness "$COLLECT_RANDOMNESS"
            --wt-path "$checkpoint_path"
            "${COLLECT_FLAGS[@]}"
            --rollout-after-success "$rollout_after_success"
        )

        log_info "Collecting data for task: ${task}"
        run_python_command "$local_root" "${collect_cmd[@]}"
    done
}

process_pickles_step() {
    local local_root="$1"
    local -a task_episode_limit_args
    local -a process_cmd

    mapfile -t task_episode_limit_args < <(build_task_episode_limit_args)
    check_process_runtime_dependencies "$local_root"

    process_cmd=(
        python -m src.data_processing.process_pickles_to_lmdb
        -c "$PROCESS_CONTROLLER"
        -d "$PROCESS_DOMAIN"
        -f "${TASKS[@]}"
        -s "$PROCESS_SOURCE"
        -r "$PROCESS_RANDOMNESS"
        -o "$PROCESS_OUTCOME"
        --suffix "$PROCESS_SUFFIX"
        --output-suffix "$PROCESS_OUTPUT_SUFFIX"
        --batch-size "$PROCESS_BATCH_SIZE"
        --task-episode-limit "${task_episode_limit_args[@]}"
        "${PROCESS_FLAGS[@]}"
    )

    log_info "Processing pickles into merged LMDB for tasks: ${TASKS[*]}"
    run_python_command "$local_root" "${process_cmd[@]}"
}

upload_step() {
    local local_root="$1"
    local dataset_upload_dir remote_ssh_host remote_dataset_dir sanitized_ld_library_path
    local ssh_bin rsync_bin
    local remote_mkdir_cmd
    local remote_probe_cmd
    local -a ssh_mkdir_cmd
    local -a ssh_probe_cmd
    local -a rsync_upload_cmd
    local rsync_ssh_cmd
    local -a ssh_common_args

    dataset_upload_dir="$(get_processed_dataset_absolute_path "$local_root")"
    [[ -d "$dataset_upload_dir" ]] || die "Merged LMDB directory does not exist: ${dataset_upload_dir}"

    remote_ssh_host="$(normalize_remote_ssh_host "${REMOTE_SSH_HOST:-}")"
    remote_dataset_dir="${REMOTE_PATH%/}/$(get_processed_dataset_relative_path)"

    [[ -n "$remote_ssh_host" ]] || die "REMOTE_SSH_HOST is required for upload."

    ssh_bin="$(resolve_command_path ssh)"
    rsync_bin="$(resolve_command_path rsync)"
    sanitized_ld_library_path="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"
    ssh_common_args=(
        -o BatchMode=yes
        -o StrictHostKeyChecking="$SSH_STRICT_HOST_KEY_CHECKING"
        -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS"
        -o ServerAliveInterval="$SSH_SERVER_ALIVE_INTERVAL_SECONDS"
        -o ServerAliveCountMax="$SSH_SERVER_ALIVE_COUNT_MAX"
        -o TCPKeepAlive=yes
        -o IPQoS=throughput
    )
    rsync_ssh_cmd="$(quote_command "$ssh_bin" "${ssh_common_args[@]}")"

    if [[ "$remote_dataset_dir" == "~/"* ]]; then
        remote_mkdir_cmd="mkdir -p -- ~/${remote_dataset_dir#~/}"
    else
        printf -v remote_mkdir_cmd 'mkdir -p -- %q' "$remote_dataset_dir"
    fi
    printf -v remote_probe_cmd 'command -v rsync >/dev/null 2>&1 && test -d %q && test -w %q' "${REMOTE_PATH%/}" "${REMOTE_PATH%/}"

    ssh_mkdir_cmd=(
        env
        "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
        "$ssh_bin"
        "${ssh_common_args[@]}"
        "$remote_ssh_host"
        "$remote_mkdir_cmd"
    )
    ssh_probe_cmd=(
        env
        "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
        "$ssh_bin"
        "${ssh_common_args[@]}"
        "$remote_ssh_host"
        "$remote_probe_cmd"
    )
    rsync_upload_cmd=(
        env
        "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
        "$rsync_bin"
        -a
        --no-owner
        --no-group
        --partial
        --partial-dir=.rsync-partial
        --human-readable
        --info=progress2
        -e
        "$rsync_ssh_cmd"
        "${dataset_upload_dir}/"
        "${remote_ssh_host}:${remote_dataset_dir}/"
    )
    if [[ -n "${UPLOAD_BWLIMIT// }" && "${UPLOAD_BWLIMIT}" != "0" ]]; then
        rsync_upload_cmd+=( "--bwlimit=${UPLOAD_BWLIMIT}" )
    fi

    run_with_retry \
        "$UPLOAD_MAX_RETRIES" \
        "$UPLOAD_RETRY_DELAY_SECONDS" \
        "Checking remote upload prerequisites on ${remote_ssh_host}" \
        "${ssh_probe_cmd[@]}"

    run_with_retry \
        "$UPLOAD_MAX_RETRIES" \
        "$UPLOAD_RETRY_DELAY_SECONDS" \
        "Ensuring remote upload directory exists: ${remote_ssh_host}:${remote_dataset_dir}" \
        "${ssh_mkdir_cmd[@]}"

    run_with_retry \
        "$UPLOAD_MAX_RETRIES" \
        "$UPLOAD_RETRY_DELAY_SECONDS" \
        "Uploading merged LMDB ${dataset_upload_dir} to ${remote_ssh_host}:${remote_dataset_dir} via rsync" \
        "${rsync_upload_cmd[@]}"
}

main() {
    local local_root step

    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"

    local_root="$(expand_path "$LOCAL_PATH")"
    [[ -d "$local_root" ]] || die "LOCAL_PATH does not exist: ${local_root}"
    require_tasks_configured

    if [[ ${#STEPS[@]} -eq 0 ]]; then
        log_info "No steps selected in STEPS; nothing to do."
        return 0
    fi

    for step in "${STEPS[@]}"; do
        case "$step" in
            collect_data)
                collect_data_step "$local_root"
                ;;
            process_pickles)
                process_pickles_step "$local_root"
                ;;
            upload)
                upload_step "$local_root"
                ;;
            *)
                die "Unknown step in STEPS: ${step}"
                ;;
        esac
    done

    log_info "auto_data_preparation finished successfully."
}

main "$@"
