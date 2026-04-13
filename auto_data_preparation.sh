#!/usr/bin/env bash

set -euo pipefail

# Comment out a line to skip that step.
STEPS=(
    # collect_data
    # process_pickles
    upload
)

LOCAL_PATH="/data/hy/robust-rearrangement"  # 218
# LOCAL_PATH="~/projects/robust-rearrangement-custom"  # base
# REMOTE_PATH="/data/hy/robust-rearrangement-custom/"  # server local
REMOTE_PATH="~/robust-rearrangement-custom/"  # server local home, for 236
# REMOTE_PATH="/mnt/nas/share/home/hy/robust-rearrangement-custom/"  # NAS
REMOTE_SSH_HOST="236"
CONDA_ENV="rr"
CONNECT_TIMEOUT_SECONDS=10
UPLOAD_MAX_RETRIES=5
UPLOAD_RETRY_DELAY_SECONDS=5

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
COLLECT_N_ROLLOUTS=44  # 要多少数据
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

PROCESS_FLAGS=(
    --overwrite
)

UPLOAD_RELATIVE_DIR="data/processed/diffik/sim"

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

run_python_command() {
    local local_root="$1"
    shift
    local -a cmd=( "$@" )

    log_info "Running command in ${local_root}: $(quote_command "${cmd[@]}")"
    (
        cd "$local_root"
        activate_conda_env "$CONDA_ENV"
        "${cmd[@]}"
    )
}

collect_data_step() {
    local local_root="$1"
    local task checkpoint_path max_rollout_steps rollout_after_success
    local -a collect_cmd

    require_command python3

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

    require_command python3

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

        log_info "Processing pickles for task: ${task}"
        run_python_command "$local_root" "${process_cmd[@]}"
    done
}

upload_step() {
    local local_root="$1"
    local upload_base_dir remote_ssh_host remote_upload_base_dir
    local task task_upload_dir remote_task_dir
    local remote_mkdir_cmd
    local -a ssh_mkdir_cmd
    local -a rsync_upload_cmd
    local rsync_ssh_cmd
    local uploaded_any

    upload_base_dir="${local_root%/}/${UPLOAD_RELATIVE_DIR}"
    [[ -d "$upload_base_dir" ]] || die "Upload directory does not exist: ${upload_base_dir}"

    remote_ssh_host="$(normalize_remote_ssh_host "${REMOTE_SSH_HOST:-}")"
    remote_upload_base_dir="${REMOTE_PATH%/}/${UPLOAD_RELATIVE_DIR}"

    [[ -n "$remote_ssh_host" ]] || die "REMOTE_SSH_HOST is required for upload."

    require_command ssh
    require_command rsync
    rsync_ssh_cmd="ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS} -o ServerAliveInterval=5 -o ServerAliveCountMax=3"

    uploaded_any=false
    for task in "${TASKS[@]}"; do
        task_upload_dir="${upload_base_dir%/}/${task}"
        if [[ ! -d "$task_upload_dir" ]]; then
            log_error "Task upload directory does not exist, skipping: ${task_upload_dir}"
            continue
        fi

        remote_task_dir="${remote_upload_base_dir%/}/${task}"
        if [[ "$remote_task_dir" == "~/"* ]]; then
            remote_mkdir_cmd="mkdir -p -- ~/${remote_task_dir#~/}"
        else
            printf -v remote_mkdir_cmd 'mkdir -p -- %q' "$remote_task_dir"
        fi
        ssh_mkdir_cmd=(
            ssh
            -o BatchMode=yes \
            -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
            -o ServerAliveInterval=5
            -o ServerAliveCountMax=3
            "$remote_ssh_host"
            "$remote_mkdir_cmd"
        )
        rsync_upload_cmd=(
            rsync
            -a
            --no-owner
            --no-group
            --partial-dir=.rsync-partial
            --human-readable
            --info=progress2
            -e
            "$rsync_ssh_cmd"
            "${task_upload_dir}/"
            "${remote_ssh_host}:${remote_task_dir}/"
        )

        run_with_retry \
            "$UPLOAD_MAX_RETRIES" \
            "$UPLOAD_RETRY_DELAY_SECONDS" \
            "Ensuring remote upload directory exists: ${remote_ssh_host}:${remote_task_dir}" \
            "${ssh_mkdir_cmd[@]}"

        run_with_retry \
            "$UPLOAD_MAX_RETRIES" \
            "$UPLOAD_RETRY_DELAY_SECONDS" \
            "Uploading folder ${task_upload_dir} to ${remote_ssh_host}:${remote_task_dir} via rsync" \
            "${rsync_upload_cmd[@]}"

        uploaded_any=true
    done

    if [[ "$uploaded_any" != true ]]; then
        die "No task upload directories found under ${upload_base_dir}"
    fi
}

main() {
    local local_root step

    local_root="$(expand_path "$LOCAL_PATH")"
    [[ -d "$local_root" ]] || die "LOCAL_PATH does not exist: ${local_root}"

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
