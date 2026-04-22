#!/usr/bin/env bash

set -euo pipefail

# User-editable configuration

# Comment out a line to skip that step.
STEPS=(
    collect_data
    process_pickles
    # upload
)

# LOCAL_PATH="/data/hy/robust-rearrangement"  # 218
LOCAL_PATH="~/projects/robust-rearrangement-custom"  # base
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
COLLECT_N_ROLLOUTS=4  # 要多少数据
COLLECT_IF_EXISTS="append"
COLLECT_ACTION_TYPE="pos"
COLLECT_OBSERVATION_SPACE="image"
COLLECT_RANDOMNESS="low"

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
PYTHON_RUNTIME_CACHE_ROOT="${PYTHON_RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/gpu-snatcher-auto-data-preparation-zarr}"

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

get_task_upload_relative_dir() {
    local task="$1"
    printf '%s/%s\n' "${UPLOAD_RELATIVE_DIR%/}" "$task"
}

get_task_upload_absolute_dir() {
    local local_root="$1"
    local task="$2"
    printf '%s/%s\n' "${local_root%/}" "$(get_task_upload_relative_dir "$task")"
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

get_conda_env_prefix() {
    local conda_exe="$1"
    local env_name="$2"
    local conda_root candidate

    conda_root="$(cd "$(dirname "$conda_exe")/.." && pwd -P)"

    if [[ "$env_name" == /* ]]; then
        candidate="$env_name"
    elif [[ "$env_name" == "base" ]]; then
        candidate="$conda_root"
    else
        candidate="${conda_root%/}/envs/${env_name}"
    fi

    if [[ -d "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    candidate="$("$conda_exe" env list | awk -v env_name="$env_name" '$1 == env_name {print $NF; exit}')"
    [[ -n "$candidate" && -d "$candidate" ]] || die "Unable to locate prefix for conda env: ${env_name}"
    printf '%s\n' "$candidate"
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
    local conda_exe conda_env_prefix conda_env_lib_path sanitized_ld_library_path python_ld_library_path
    local -a cmd=( "$@" )

    conda_exe="$(get_conda_executable)"
    conda_env_prefix="$(get_conda_env_prefix "$conda_exe" "$CONDA_ENV")"
    conda_env_lib_path="${conda_env_prefix%/}/lib"
    [[ -d "$conda_env_lib_path" ]] || die "Conda env lib directory does not exist: ${conda_env_lib_path}"
    ensure_python_runtime_dirs "$PYTHON_RUNTIME_CACHE_ROOT"
    sanitized_ld_library_path="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"
    python_ld_library_path="$conda_env_lib_path"
    if [[ -n "$sanitized_ld_library_path" ]]; then
        python_ld_library_path="${python_ld_library_path}:${sanitized_ld_library_path}"
    fi

    log_info "Running command in ${local_root} with conda env ${CONDA_ENV}: $(quote_command "${cmd[@]}")"
    env \
        LD_LIBRARY_PATH="$python_ld_library_path" \
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

required_modules = ("zarr",)
failures = []

for module_name in required_modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - shell preflight
        failures.append((module_name, exc, traceback.format_exc()))

if failures:
    print("Python dependency check failed before process_pickles (Zarr flow).", file=sys.stderr)
    for module_name, exc, tb in failures:
        print(f"[dependency] {module_name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
    print(
        "Reinstall the broken package inside the target env, for example: "
        "conda run -n rr python -m pip install --force-reinstall zarr",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

build_remote_mkdir_cmd() {
    local remote_dir="$1"

    if [[ "$remote_dir" == "~" ]]; then
        printf '%s\n' "mkdir -p -- ~"
    elif [[ "$remote_dir" == "~/"* ]]; then
        printf '%s\n' "mkdir -p -- ~/${remote_dir#~/}"
    else
        printf 'mkdir -p -- %q\n' "$remote_dir"
    fi
}

build_remote_probe_cmd() {
    local remote_base_dir="$1"

    if [[ "$remote_base_dir" == "~" ]]; then
        printf '%s\n' "command -v rsync >/dev/null 2>&1 && test -d ~ && test -w ~"
    elif [[ "$remote_base_dir" == "~/"* ]]; then
        printf '%s\n' "command -v rsync >/dev/null 2>&1 && test -d ~/${remote_base_dir#~/} && test -w ~/${remote_base_dir#~/}"
    else
        printf 'command -v rsync >/dev/null 2>&1 && test -d %q && test -w %q\n' "$remote_base_dir" "$remote_base_dir"
    fi
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
    local task
    local -a process_cmd

    check_process_runtime_dependencies "$local_root"

    for task in "${TASKS[@]}"; do
        process_cmd=(
            python -m src.data_processing.process_pickles
            -c "$PROCESS_CONTROLLER"
            -d "$PROCESS_DOMAIN"
            -f "$task"
            -s "$PROCESS_SOURCE"
            -r "$PROCESS_RANDOMNESS"
            -o "$PROCESS_OUTCOME"
            --suffix "$PROCESS_SUFFIX"
            --output-suffix "$PROCESS_OUTPUT_SUFFIX"
            --batch-size "$PROCESS_BATCH_SIZE"
            "${PROCESS_FLAGS[@]}"
        )

        log_info "Processing pickles into task-local Zarr outputs for task: ${task}"
        run_python_command "$local_root" "${process_cmd[@]}"
    done
}

try_direct_nas_rsync() {
    local source_dir="$1"
    local target_dir="$2"
    local rsync_bin
    local status
    local -a direct_rsync_cmd

    log_info "REMOTE_PATH is under /mnt/nas; trying direct local rsync before SSH rsync."
    log_info "Direct rsync source: ${source_dir}"
    log_info "Direct rsync target: ${target_dir}"

    rsync_bin="$(command -v rsync 2>/dev/null || true)"
    if [[ -z "$rsync_bin" ]]; then
        log_info "Local rsync command not found; falling back to SSH rsync."
        return 1
    fi

    log_info "Ensuring direct rsync target directory exists: ${target_dir}"
    if mkdir -p "$target_dir"; then
        log_info "Direct rsync target directory is ready: ${target_dir}"
    else
        status=$?
        log_info "Direct rsync mkdir failed with exit code ${status}; falling back to SSH rsync."
        return "$status"
    fi

    direct_rsync_cmd=(
        "$rsync_bin"
        -a
        --no-owner
        --no-group
        --partial
        --partial-dir=.rsync-partial
        --human-readable
        --info=progress2
    )
    if [[ -n "${UPLOAD_BWLIMIT// }" && "${UPLOAD_BWLIMIT}" != "0" ]]; then
        direct_rsync_cmd+=( "--bwlimit=${UPLOAD_BWLIMIT}" )
    fi
    direct_rsync_cmd+=( "${source_dir}/" "${target_dir}/" )

    log_info "Copying task dataset tree to mounted NAS with local rsync progress. This can take a while for large datasets."
    if "${direct_rsync_cmd[@]}"; then
        log_info "Direct NAS rsync finished successfully: ${target_dir}"
        return 0
    else
        status=$?
        log_info "Direct NAS rsync failed with exit code ${status}; falling back to SSH rsync."
        return "$status"
    fi
}

upload_step() {
    local local_root="$1"
    local task task_upload_dir remote_task_dir remote_ssh_host sanitized_ld_library_path
    local ssh_bin rsync_bin
    local remote_mkdir_cmd remote_probe_cmd
    local rsync_ssh_cmd
    local uploaded_any remote_prerequisites_checked
    local -a ssh_common_args=()
    local -a ssh_mkdir_cmd
    local -a ssh_probe_cmd
    local -a rsync_upload_cmd

    sanitized_ld_library_path="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"
    remote_ssh_host=""
    ssh_bin=""
    rsync_bin=""
    remote_probe_cmd=""
    rsync_ssh_cmd=""
    uploaded_any=false
    remote_prerequisites_checked=false

    for task in "${TASKS[@]}"; do
        task_upload_dir="$(get_task_upload_absolute_dir "$local_root" "$task")"
        if [[ ! -d "$task_upload_dir" ]]; then
            log_error "Task upload directory does not exist, skipping: ${task_upload_dir}"
            continue
        fi

        remote_task_dir="${REMOTE_PATH%/}/$(get_task_upload_relative_dir "$task")"

        if [[ "$remote_task_dir" == /mnt/nas* ]]; then
            if try_direct_nas_rsync "$task_upload_dir" "$remote_task_dir"; then
                uploaded_any=true
                continue
            fi
        fi

        if [[ -z "$remote_ssh_host" ]]; then
            remote_ssh_host="$(normalize_remote_ssh_host "${REMOTE_SSH_HOST:-}")"
            [[ -n "$remote_ssh_host" ]] || die "REMOTE_SSH_HOST is required for upload when direct NAS rsync is unavailable."

            ssh_bin="$(resolve_command_path ssh)"
            rsync_bin="$(resolve_command_path rsync)"
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
            remote_probe_cmd="$(build_remote_probe_cmd "${REMOTE_PATH%/}")"
        fi

        if [[ "$remote_prerequisites_checked" != true ]]; then
            ssh_probe_cmd=(
                env
                "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
                "$ssh_bin"
                "${ssh_common_args[@]}"
                "$remote_ssh_host"
                "$remote_probe_cmd"
            )

            run_with_retry \
                "$UPLOAD_MAX_RETRIES" \
                "$UPLOAD_RETRY_DELAY_SECONDS" \
                "Checking remote upload prerequisites on ${remote_ssh_host}" \
                "${ssh_probe_cmd[@]}"

            remote_prerequisites_checked=true
        fi

        remote_mkdir_cmd="$(build_remote_mkdir_cmd "$remote_task_dir")"
        ssh_mkdir_cmd=(
            env
            "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
            "$ssh_bin"
            "${ssh_common_args[@]}"
            "$remote_ssh_host"
            "$remote_mkdir_cmd"
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
            "${task_upload_dir}/"
            "${remote_ssh_host}:${remote_task_dir}/"
        )
        if [[ -n "${UPLOAD_BWLIMIT// }" && "${UPLOAD_BWLIMIT}" != "0" ]]; then
            rsync_upload_cmd+=( "--bwlimit=${UPLOAD_BWLIMIT}" )
        fi

        run_with_retry \
            "$UPLOAD_MAX_RETRIES" \
            "$UPLOAD_RETRY_DELAY_SECONDS" \
            "Ensuring remote upload directory exists: ${remote_ssh_host}:${remote_task_dir}" \
            "${ssh_mkdir_cmd[@]}"

        run_with_retry \
            "$UPLOAD_MAX_RETRIES" \
            "$UPLOAD_RETRY_DELAY_SECONDS" \
            "Uploading task dataset ${task_upload_dir} to ${remote_ssh_host}:${remote_task_dir} via rsync" \
            "${rsync_upload_cmd[@]}"

        uploaded_any=true
    done

    if [[ "$uploaded_any" != true ]]; then
        die "No task upload directories found under ${local_root%/}/${UPLOAD_RELATIVE_DIR}"
    fi
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

    log_info "auto_data_preparation_zarr finished successfully."
}

main "$@"
