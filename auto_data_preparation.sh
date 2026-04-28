#!/usr/bin/env bash

set -euo pipefail

# User-editable configuration

# Comment out a line to skip that step.
STEPS=(
    # collect_data
    # process_pickles
    upload
)

# LOCAL_PATH="/data/hy/robust-rearrangement"  # 218
LOCAL_PATH="~/projects/robust-rearrangement-custom"  # base
REMOTE_PATH="/data/hy/robust-rearrangement-custom/"  # server local
# REMOTE_PATH="~/robust-rearrangement-custom/"  # server local home, for 236
# REMOTE_PATH="/mnt/nas/share/home/hy/robust-rearrangement-custom/"  # NAS
REMOTE_SSH_HOST="240"
CONDA_ENV="rr"
CONNECT_TIMEOUT_SECONDS=10
UPLOAD_MAX_RETRIES=5
UPLOAD_RETRY_DELAY_SECONDS=5
SSH_STRICT_HOST_KEY_CHECKING="${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
SSH_SERVER_ALIVE_INTERVAL_SECONDS="${SSH_SERVER_ALIVE_INTERVAL_SECONDS:-15}"
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-12}"
UPLOAD_BWLIMIT="${UPLOAD_BWLIMIT:-100m}"
split_file=true
part_size=1024  # 单位：MB
parallel_upload_workers=4

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
COLLECT_ANNOTATE_SKILL=false  # 是否加 2d guidance point 和收集 skill 标注
COLLECT_SKILL_ON_IMAGE=false

# 等比例配置数据
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
    --output-only-pickle
)
if [[ "$COLLECT_ANNOTATE_SKILL" == "true" ]]; then
    COLLECT_FLAGS+=(--annotate-skill)
fi
if [[ "$COLLECT_ANNOTATE_SKILL" == "true" && "$COLLECT_SKILL_ON_IMAGE" == "true" ]]; then
    COLLECT_FLAGS+=(--skill-on-image)
fi

PROCESS_CONTROLLER="diffik"
PROCESS_DOMAIN="sim"
PROCESS_SOURCE="rollout"
PROCESS_RANDOMNESS="low"
PROCESS_OUTCOME="success"
PROCESS_SUFFIX="$([[ "$COLLECT_ANNOTATE_SKILL" == "true" ]] && printf 'rgbd-skill' || printf 'rgbd')"
PROCESS_OUTPUT_SUFFIX="$PROCESS_SUFFIX"
PROCESS_BATCH_SIZE=2
PYTHON_RUNTIME_CACHE_ROOT="${PYTHON_RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/gpu-snatcher-auto-data-preparation}"

PROCESS_FLAGS=(
    --overwrite
)

UPLOAD_RELATIVE_DIR="data/processed/diffik/sim"

SPLIT_FILE_ENABLED=false
PART_SIZE_BYTES=0
PARALLEL_UPLOAD_WORKERS=0

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

validate_upload_config() {
    case "$split_file" in
        true|false)
            ;;
        *)
            die "split_file must be true or false, got: ${split_file}"
            ;;
    esac

    [[ "$part_size" =~ ^[0-9]+$ ]] || die "part_size must be a positive integer in MB, got: ${part_size}"
    (( part_size > 0 )) || die "part_size must be greater than 0 MB, got: ${part_size}"

    [[ "$parallel_upload_workers" =~ ^[0-9]+$ ]] || die "parallel_upload_workers must be a positive integer, got: ${parallel_upload_workers}"
    (( parallel_upload_workers > 0 )) || die "parallel_upload_workers must be greater than 0, got: ${parallel_upload_workers}"

    SPLIT_FILE_ENABLED="$split_file"
    PART_SIZE_BYTES=$(( part_size * 1024 * 1024 ))
    PARALLEL_UPLOAD_WORKERS="$parallel_upload_workers"
}

ensure_split_upload_runtime_dependencies() {
    require_command dd
    require_command find
    require_command sort
    require_command touch
    require_command tr
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

    case "$remote_base_dir" in
        "~")
            printf '%s\n' "command -v rsync >/dev/null 2>&1 && command -v cat >/dev/null 2>&1 && command -v wc >/dev/null 2>&1 && command -v mv >/dev/null 2>&1 && command -v rm >/dev/null 2>&1 && command -v mkdir >/dev/null 2>&1 && test -d ~ && test -w ~"
            ;;
        "~/"*)
            printf '%s\n' "command -v rsync >/dev/null 2>&1 && command -v cat >/dev/null 2>&1 && command -v wc >/dev/null 2>&1 && command -v mv >/dev/null 2>&1 && command -v rm >/dev/null 2>&1 && command -v mkdir >/dev/null 2>&1 && test -d ~/${remote_base_dir#~/} && test -w ~/${remote_base_dir#~/}"
            ;;
        *)
            printf 'command -v rsync >/dev/null 2>&1 && command -v cat >/dev/null 2>&1 && command -v wc >/dev/null 2>&1 && command -v mv >/dev/null 2>&1 && command -v rm >/dev/null 2>&1 && command -v mkdir >/dev/null 2>&1 && test -d %q && test -w %q\n' "$remote_base_dir" "$remote_base_dir"
            ;;
    esac
}

sanitize_split_upload_key() {
    local value="$1"

    value="${value//\//_}"
    value="${value// /_}"
    value="${value//$'\t'/_}"
    value="${value//$'\n'/_}"
    value="${value//$'\r'/_}"
    printf '%s' "$value" | tr -c 'A-Za-z0-9._-' '_'
}

collect_split_file_candidates() {
    local source_dir="$1"
    local part_size_bytes="$2"
    local -n split_paths_ref="$3"
    local -n split_sizes_ref="$4"
    local relative_path file_size

    split_paths_ref=()
    split_sizes_ref=()

    while IFS=$'\t' read -r -d '' relative_path file_size; do
        [[ -n "$relative_path" ]] || continue
        split_paths_ref+=("$relative_path")
        split_sizes_ref+=("$file_size")
    done < <(find "$source_dir" -type f -size +"${part_size_bytes}"c -printf '%P\t%s\0' | LC_ALL=C sort -z)
}

get_remote_dataset_file_path() {
    local remote_dataset_dir="$1"
    local relative_path="$2"
    printf '%s/%s\n' "${remote_dataset_dir%/}" "$relative_path"
}

get_remote_dataset_parent_dir() {
    local remote_dataset_dir="$1"
    local relative_path="$2"

    if [[ "$relative_path" == */* ]]; then
        printf '%s/%s\n' "${remote_dataset_dir%/}" "${relative_path%/*}"
    else
        printf '%s\n' "$remote_dataset_dir"
    fi
}

format_remote_shell_value() {
    local value="$1"

    if [[ "$value" == "~" || "$value" == "~/"* ]]; then
        printf '%s' "$value"
    else
        printf '%q' "$value"
    fi
}

build_remote_merge_cmd() {
    local remote_staging_dir="$1"
    local remote_final_file="$2"
    local expected_part_count="$3"
    local expected_file_size="$4"
    local remote_uploading_file script

    remote_uploading_file="${remote_final_file}.uploading"

    script="set -eu;"
    script+=" LC_ALL=C; export LC_ALL;"
    script+=" staging_dir=$(format_remote_shell_value "$remote_staging_dir");"
    script+=" final_file=$(format_remote_shell_value "$remote_final_file");"
    script+=" uploading_file=$(format_remote_shell_value "$remote_uploading_file");"
    script+=" expected_part_count=${expected_part_count};"
    script+=" expected_file_size=${expected_file_size};"
    script+=" cd \"\$staging_dir\";"
    script+=" part_number=1;"
    script+=" part_number_width=${#expected_part_count};"
    script+=" rm -f -- \"\$uploading_file\";"
    script+=" : > \"\$uploading_file\";"
    script+=" while [ \"\$part_number\" -le \"\$expected_part_count\" ]; do"
    script+=" part_name=\$(printf 'part-%0*d' \"\$part_number_width\" \"\$part_number\");"
    script+=" if [ ! -f \"\$part_name\" ]; then echo \"Missing expected split part \$part_name in \$staging_dir\" >&2; rm -f -- \"\$uploading_file\"; exit 1; fi;"
    script+=" cat \"\$part_name\" >> \"\$uploading_file\";"
    script+=" part_number=\$((part_number + 1));"
    script+=" done;"
    script+=" actual_file_size=\$(wc -c < \"\$uploading_file\");"
    script+=" set -- \$actual_file_size;"
    script+=" actual_file_size=\$1;"
    script+=" if [ \"\$actual_file_size\" -ne \"\$expected_file_size\" ]; then echo \"Merged file size mismatch for \$final_file: expected \$expected_file_size, got \$actual_file_size\" >&2; rm -f -- \"\$uploading_file\"; exit 1; fi;"
    script+=" mv -f -- \"\$uploading_file\" \"\$final_file\";"
    script+=" rm -rf -- \"\$staging_dir\""

    printf 'sh -eu -c %q\n' "$script"
}

wait_for_upload_slot() {
    local max_parallel_jobs="$1"
    local -n active_jobs_ref="$2"
    local status

    while (( active_jobs_ref >= max_parallel_jobs )); do
        if wait -n; then
            active_jobs_ref=$((active_jobs_ref - 1))
        else
            status=$?
            active_jobs_ref=$((active_jobs_ref - 1))
            return "$status"
        fi
    done
}

wait_for_all_upload_jobs() {
    local -n active_jobs_ref="$1"
    local status

    while (( active_jobs_ref > 0 )); do
        if wait -n; then
            active_jobs_ref=$((active_jobs_ref - 1))
        else
            status=$?
            active_jobs_ref=$((active_jobs_ref - 1))
            while (( active_jobs_ref > 0 )); do
                wait -n || true
                active_jobs_ref=$((active_jobs_ref - 1))
            done
            return "$status"
        fi
    done
}

upload_split_part() {
    local source_file="$1"
    local relative_path="$2"
    local source_file_size="$3"
    local part_size_bytes="$4"
    local part_number="$5"
    local total_parts="$6"
    local local_temp_root="$7"
    local remote_staging_dir="$8"
    local remote_ssh_host="$9"
    local sanitized_ld_library_path="${10}"
    local rsync_bin="${11}"
    local rsync_ssh_cmd="${12}"
    local part_number_width part_name local_part_file part_offset remaining_bytes current_part_bytes
    local -a rsync_part_cmd

    mkdir -p "$local_temp_root"
    part_number_width="${#total_parts}"
    printf -v part_name 'part-%0*d' "$part_number_width" "$part_number"
    local_part_file="${local_temp_root%/}/${part_name}"
    trap 'rm -f -- "$local_part_file"' EXIT

    part_offset=$(( (part_number - 1) * part_size_bytes ))
    remaining_bytes=$(( source_file_size - part_offset ))
    if (( remaining_bytes <= part_size_bytes )); then
        current_part_bytes="$remaining_bytes"
    else
        current_part_bytes="$part_size_bytes"
    fi

    dd \
        if="$source_file" \
        of="$local_part_file" \
        bs=4M \
        iflag=skip_bytes,count_bytes,fullblock \
        skip="$part_offset" \
        count="$current_part_bytes" \
        status=none
    touch -r "$source_file" "$local_part_file"

    rsync_part_cmd=(
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
    )
    if [[ -n "${UPLOAD_BWLIMIT// }" && "${UPLOAD_BWLIMIT}" != "0" ]]; then
        rsync_part_cmd+=( "--bwlimit=${UPLOAD_BWLIMIT}" )
    fi
    rsync_part_cmd+=(
        "$local_part_file"
        "${remote_ssh_host}:${remote_staging_dir%/}/${part_name}"
    )

    run_with_retry \
        "$UPLOAD_MAX_RETRIES" \
        "$UPLOAD_RETRY_DELAY_SECONDS" \
        "Uploading split part ${part_number}/${total_parts} for ${relative_path}" \
        "${rsync_part_cmd[@]}"
}

upload_large_file_in_parts() {
    local dataset_upload_dir="$1"
    local relative_path="$2"
    local source_file_size="$3"
    local remote_dataset_dir="$4"
    local remote_ssh_host="$5"
    local sanitized_ld_library_path="$6"
    local ssh_bin="$7"
    local rsync_bin="$8"
    local rsync_ssh_cmd="$9"
    local source_file remote_final_file remote_parent_dir split_key local_temp_root remote_staging_dir
    local remote_setup_cmd remote_merge_cmd
    local -a ssh_setup_cmd ssh_merge_cmd
    local total_parts active_jobs
    local part_number

    source_file="${dataset_upload_dir%/}/${relative_path}"
    [[ -f "$source_file" ]] || die "Split upload source file does not exist: ${source_file}"

    total_parts=$(( (source_file_size + PART_SIZE_BYTES - 1) / PART_SIZE_BYTES ))
    remote_final_file="$(get_remote_dataset_file_path "$remote_dataset_dir" "$relative_path")"
    remote_parent_dir="$(get_remote_dataset_parent_dir "$remote_dataset_dir" "$relative_path")"
    split_key="$(sanitize_split_upload_key "${relative_path}-${source_file_size}-${PART_SIZE_BYTES}")"
    local_temp_root="${TMPDIR:-/tmp}/gpu-snatcher-split-upload/${split_key}"
    remote_staging_dir="${remote_dataset_dir%/}/.split-upload-${split_key}.parts"

    log_info "Uploading large file in ${total_parts} parts: ${relative_path} (${source_file_size} bytes, ${PART_SIZE_BYTES} bytes per part, ${PARALLEL_UPLOAD_WORKERS} workers)"

    mkdir -p "$local_temp_root"
    remote_setup_cmd="$(build_remote_mkdir_cmd "$remote_parent_dir") && $(build_remote_mkdir_cmd "$remote_staging_dir")"
    ssh_setup_cmd=(
        env
        "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
        "$ssh_bin"
        -o BatchMode=yes
        -o StrictHostKeyChecking="$SSH_STRICT_HOST_KEY_CHECKING"
        -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS"
        -o ServerAliveInterval="$SSH_SERVER_ALIVE_INTERVAL_SECONDS"
        -o ServerAliveCountMax="$SSH_SERVER_ALIVE_COUNT_MAX"
        -o TCPKeepAlive=yes
        -o IPQoS=throughput
        "$remote_ssh_host"
        "$remote_setup_cmd"
    )
    run_with_retry \
        "$UPLOAD_MAX_RETRIES" \
        "$UPLOAD_RETRY_DELAY_SECONDS" \
        "Ensuring remote split upload staging exists for ${relative_path}" \
        "${ssh_setup_cmd[@]}"

    active_jobs=0
    for (( part_number = 1; part_number <= total_parts; part_number++ )); do
        wait_for_upload_slot "$PARALLEL_UPLOAD_WORKERS" active_jobs || {
            wait_for_all_upload_jobs active_jobs || true
            return 1
        }
        upload_split_part \
            "$source_file" \
            "$relative_path" \
            "$source_file_size" \
            "$PART_SIZE_BYTES" \
            "$part_number" \
            "$total_parts" \
            "$local_temp_root" \
            "$remote_staging_dir" \
            "$remote_ssh_host" \
            "$sanitized_ld_library_path" \
            "$rsync_bin" \
            "$rsync_ssh_cmd" &
        active_jobs=$((active_jobs + 1))
    done
    wait_for_all_upload_jobs active_jobs

    remote_merge_cmd="$(build_remote_merge_cmd "$remote_staging_dir" "$remote_final_file" "$total_parts" "$source_file_size")"
    ssh_merge_cmd=(
        env
        "LD_LIBRARY_PATH=${sanitized_ld_library_path}"
        "$ssh_bin"
        -o BatchMode=yes
        -o StrictHostKeyChecking="$SSH_STRICT_HOST_KEY_CHECKING"
        -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS"
        -o ServerAliveInterval="$SSH_SERVER_ALIVE_INTERVAL_SECONDS"
        -o ServerAliveCountMax="$SSH_SERVER_ALIVE_COUNT_MAX"
        -o TCPKeepAlive=yes
        -o IPQoS=throughput
        "$remote_ssh_host"
        "$remote_merge_cmd"
    )
    run_with_retry \
        "$UPLOAD_MAX_RETRIES" \
        "$UPLOAD_RETRY_DELAY_SECONDS" \
        "Merging split parts for ${relative_path} on ${remote_ssh_host}" \
        "${ssh_merge_cmd[@]}"

    rmdir "$local_temp_root" 2>/dev/null || true
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

    log_info "Copying merged LMDB to mounted NAS with local rsync progress. This can take a while for large datasets."
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
    local dataset_upload_dir remote_ssh_host remote_dataset_dir sanitized_ld_library_path
    local ssh_bin rsync_bin
    local remote_mkdir_cmd
    local remote_probe_cmd
    local -a ssh_mkdir_cmd
    local -a ssh_probe_cmd
    local -a rsync_upload_cmd
    local rsync_ssh_cmd
    local -a ssh_common_args
    local -a split_paths=()
    local -a split_sizes=()
    local split_index

    dataset_upload_dir="$(get_processed_dataset_absolute_path "$local_root")"
    [[ -d "$dataset_upload_dir" ]] || die "Merged LMDB directory does not exist: ${dataset_upload_dir}"
    validate_upload_config

    remote_ssh_host="$(normalize_remote_ssh_host "${REMOTE_SSH_HOST:-}")"
    remote_dataset_dir="${REMOTE_PATH%/}/$(get_processed_dataset_relative_path)"

    if [[ "$remote_dataset_dir" == /mnt/nas* ]]; then
        if [[ "$SPLIT_FILE_ENABLED" == "true" ]]; then
            log_info "split_file=true is bypassed for direct /mnt/nas rsync. Split upload will only be used if SSH fallback is needed."
        fi
        if try_direct_nas_rsync "$dataset_upload_dir" "$remote_dataset_dir"; then
            return 0
        fi
    fi

    [[ -n "$remote_ssh_host" ]] || die "REMOTE_SSH_HOST is required for upload."

    if [[ "$SPLIT_FILE_ENABLED" == "true" ]]; then
        ensure_split_upload_runtime_dependencies
        collect_split_file_candidates "$dataset_upload_dir" "$PART_SIZE_BYTES" split_paths split_sizes
        if (( ${#split_paths[@]} > 0 )); then
            log_info "Split upload enabled: ${#split_paths[@]} large file(s) exceed ${PART_SIZE_BYTES} bytes."
        else
            log_info "Split upload enabled, but no files exceed ${PART_SIZE_BYTES} bytes. Falling back to the regular rsync upload path."
        fi
    fi

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

    remote_mkdir_cmd="$(build_remote_mkdir_cmd "$remote_dataset_dir")"
    remote_probe_cmd="$(build_remote_probe_cmd "${REMOTE_PATH%/}")"

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
    )
    if [[ "$SPLIT_FILE_ENABLED" == "true" ]]; then
        for split_index in "${!split_paths[@]}"; do
            rsync_upload_cmd+=( "--exclude=/${split_paths[$split_index]}" )
        done
    fi
    if [[ -n "${UPLOAD_BWLIMIT// }" && "${UPLOAD_BWLIMIT}" != "0" ]]; then
        rsync_upload_cmd+=( "--bwlimit=${UPLOAD_BWLIMIT}" )
    fi
    rsync_upload_cmd+=(
        "${dataset_upload_dir}/"
        "${remote_ssh_host}:${remote_dataset_dir}/"
    )

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

    if [[ "$SPLIT_FILE_ENABLED" == "true" ]]; then
        for split_index in "${!split_paths[@]}"; do
            upload_large_file_in_parts \
                "$dataset_upload_dir" \
                "${split_paths[$split_index]}" \
                "${split_sizes[$split_index]}" \
                "$remote_dataset_dir" \
                "$remote_ssh_host" \
                "$sanitized_ld_library_path" \
                "$ssh_bin" \
                "$rsync_bin" \
                "$rsync_ssh_cmd"
        done
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

    log_info "auto_data_preparation finished successfully."
}

main "$@"
