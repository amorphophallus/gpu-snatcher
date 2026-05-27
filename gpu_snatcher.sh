#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 超参数 —— 修改这里即可
# ============================================================

# 要监视的服务器列表（数字后缀如 230，或全名如 zju_4090_230）
# 留空表示监视 SSH config 中所有 zju_ 服务器
TARGET_SERVERS=()

# 每组需要的 GPU 数量（必须在同一台服务器上）
GPUS_PER_GROUP=2

# 总共需要占几组（占满后不再抢新 GPU，但继续监视等待用户释放）
NUM_GROUPS=3

# 目标显存占用比例（占完后该卡总占用率 ≈ 此值）
GPU_MEMORY_TARGET_RATIO=0.9

# 安全余量（MiB），防止 CUDA context / 碎片化导致 OOM
GPU_MEMORY_HEADROOM_MIB=256

# 轮询间隔（秒）
POLL_INTERVAL=30

# SSH 连接超时（秒）
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-5}"

# SSH 命令超时（秒）
SSH_COMMAND_TIMEOUT_SECONDS="${SSH_COMMAND_TIMEOUT_SECONDS:-15}"

# 内存使用阈值：低于此比例的 GPU 视为空闲（默认 0.1 = 10%）
MEMORY_USAGE_THRESHOLD="${MEMORY_USAGE_THRESHOLD:-0.1}"

# tmux 会话名前缀
TMUX_SESSION_PREFIX="gpu_snatch"

# 远程 conda 环境名（需装有 PyTorch + CUDA）
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-rr}"

# 远程工作目录（存放占显存脚本）
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/tmp}"

# SSH config 路径
SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"

# Dry run 模式：设为 1 只打印不执行（用于调试）
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# SSH 工具函数
# ============================================================

SSH_COMMON_ARGS=(
    -o BatchMode=yes
    -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=1
)

run_ssh() {
    if (( DRY_RUN )); then
        echo "[DRY_RUN] ssh ${SSH_COMMON_ARGS[*]} $*" >&2
        return 0
    fi
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

# ============================================================
# 服务器发现
# ============================================================

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

# ============================================================
# GPU 状态查询
# ============================================================

# 查询一台服务器的 GPU 状态
# 输出格式：
#   HOST|<host>|OK|
#   GPU|<host>|<index>|FREE/BUSY|<used_mib>|<total_mib>|<usage_pct>|<gpu_util_pct>
#   HOST|<host>|DOWN|<base64_error>
get_host_gpu_status() {
    local host_alias="$1"
    local query_output

    if (( DRY_RUN )); then
        # Dry run: return an empty OK response (no GPUs reported)
        printf 'HOST|%s|OK|\n' "$host_alias"
        return 0
    fi

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

# ============================================================
# GPU 占显存脚本（Python，base64 编码后传到远程执行）
# ============================================================

# 生成占显存 Python 脚本的 base64 编码
# 参数：gpu_ids(逗号分隔) target_ratio headroom_mib
build_gpu_occupy_script_b64() {
    python3 - "$@" <<'PY'
import base64
import sys
import textwrap

gpu_ids_str = sys.argv[1]
target_ratio = float(sys.argv[2])
headroom_mib = float(sys.argv[3])

script = textwrap.dedent(f'''
import torch
import sys
import time

gpu_ids_str = {repr(gpu_ids_str)}
target_ratio = {target_ratio}
headroom_mib = {headroom_mib}
headroom_bytes = int(headroom_mib * 1024 * 1024)

kept_tensors = []

for gpu_id_str in gpu_ids_str.split(","):
    gpu_id = int(gpu_id_str.strip())
    torch.cuda.set_device(gpu_id)
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)

    used_bytes = total_bytes - free_bytes
    target_bytes = int(total_bytes * target_ratio)
    needed_bytes = target_bytes - used_bytes - headroom_bytes

    used_gb = used_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    if needed_bytes <= 0:
        current_pct = used_bytes / total_bytes * 100
        print(f"GPU {{gpu_id}}: already at {{current_pct:.1f}}% ({{used_gb:.2f}}/{{total_gb:.2f}} GiB), skipping")
        sys.stdout.flush()
        continue

    max_safe = free_bytes - headroom_bytes
    alloc_bytes = min(needed_bytes, max_safe)

    if alloc_bytes <= 0:
        print(f"GPU {{gpu_id}}: insufficient free memory "
              f"(free={{free_bytes/(1024**3):.2f}} GiB, needed>{{needed_bytes/(1024**3):.2f}} GiB)")
        sys.stdout.flush()
        continue

    num_elements = alloc_bytes // 4
    allocated = False
    while not allocated and alloc_bytes > 0:
        num_elements = alloc_bytes // 4
        try:
            t = torch.zeros(num_elements, dtype=torch.float32, device=f"cuda:{{gpu_id}}")
            kept_tensors.append(t)
            _, after_free = torch.cuda.mem_get_info(gpu_id)
            actual_used = total_bytes - after_free
            print(f"GPU {{gpu_id}}: +{{alloc_bytes/(1024**3):.2f}} GiB allocated, "
                  f"{{used_gb:.2f}} -> {{actual_used/(1024**3):.2f}} GiB "
                  f"({{actual_used/total_bytes*100:.1f}}% used)")
            sys.stdout.flush()
            allocated = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                alloc_bytes = int(alloc_bytes * 0.8)
                print(f"GPU {{gpu_id}}: OOM, retrying with {{alloc_bytes/(1024**3):.2f}} GiB...")
                sys.stdout.flush()
            else:
                print(f"GPU {{gpu_id}}: error - {{e}}")
                sys.stdout.flush()
                break

if kept_tensors:
    print(f"Occupation active on {{len(kept_tensors)}} GPU(s). Waiting for termination...")
else:
    print("No tensors allocated. Still waiting to hold the session...")
sys.stdout.flush()

while True:
    time.sleep(10)
''')

print(base64.b64encode(script.encode()).decode())
PY
}

# ============================================================
# tmux 远程会话管理
# ============================================================

# 在远程服务器上启动 GPU 占显存 tmux 会话
# 参数: host_alias session_name gpu_ids_csv
launch_gpu_occupation() {
    local host_alias="$1"
    local session_name="$2"
    local gpu_ids_csv="$3"

    local occupy_b64
    occupy_b64="$(build_gpu_occupy_script_b64 "$gpu_ids_csv" "$GPU_MEMORY_TARGET_RATIO" "$GPU_MEMORY_HEADROOM_MIB")"

    if (( DRY_RUN )); then
        echo "[DRY_RUN] Would launch GPU occupation on $host_alias:"
        echo "         session=$session_name gpus=$gpu_ids_csv"
        echo "         target_ratio=$GPU_MEMORY_TARGET_RATIO headroom=${GPU_MEMORY_HEADROOM_MIB}MiB"
        return 0
    fi

    run_ssh "$host_alias" bash -s -- "$session_name" "$REMOTE_WORK_DIR" "$REMOTE_CONDA_ENV" "$occupy_b64" "$gpu_ids_csv" "$GPU_MEMORY_TARGET_RATIO" "$GPU_MEMORY_HEADROOM_MIB" <<'REMOTE'
set -euo pipefail

session_name="$1"
work_dir="$2"
conda_env="$3"
occupy_b64="$4"
gpu_ids_csv="$5"
target_ratio="$6"
headroom_mib="$7"

if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux is not installed on $(hostname)" >&2
    exit 1
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "ERROR: tmux session $session_name already exists on $(hostname)" >&2
    exit 1
fi

mkdir -p "$work_dir"

# 将 base64 脚本解码写入临时文件
occupy_script="$work_dir/gpu_occupy_${session_name}.py"
echo "$occupy_b64" | base64 -d > "$occupy_script"

tmux new-session -d -s "$session_name"
tmux set-option -t "$session_name" remain-on-exit on
tmux new-window -t "$session_name" -n occupy
tmux kill-window -t "${session_name}:0" 2>/dev/null || true

# 逐条发送命令
tmux send-keys -t "${session_name}:occupy" "cd $work_dir" Enter
tmux send-keys -t "${session_name}:occupy" "source ~/.bashrc" Enter
tmux send-keys -t "${session_name}:occupy" 'eval "$(conda shell.bash hook)"' Enter
tmux send-keys -t "${session_name}:occupy" "conda activate $conda_env" Enter
tmux send-keys -t "${session_name}:occupy" "echo 'GPU occupation starting on GPUs: $gpu_ids_csv'" Enter
tmux send-keys -t "${session_name}:occupy" "echo \"Target ratio: $target_ratio, Headroom: ${headroom_mib}MiB\"" Enter
tmux send-keys -t "${session_name}:occupy" "python3 $occupy_script" Enter

echo "Launched tmux session '$session_name' on $(hostname) for GPUs: $gpu_ids_csv"
REMOTE
}

# 释放一台服务器上的占显存会话
release_gpu_occupation() {
    local host_alias="$1"
    local session_name="$2"

    if (( DRY_RUN )); then
        echo "[DRY_RUN] Would kill tmux session $session_name on $host_alias"
        return 0
    fi

    echo "  Releasing $session_name on $host_alias..."
    run_ssh "$host_alias" bash -s -- "$session_name" <<'REMOTE'
set -euo pipefail
session_name="$1"

if tmux has-session -t "$session_name" 2>/dev/null; then
    tmux kill-session -t "$session_name"
    echo "    Released $session_name on $(hostname)"
else
    echo "    Session $session_name not found on $(hostname) (already released?)"
fi
REMOTE
}

# ============================================================
# 释放所有已占 GPU（cleanup / Ctrl+C 时调用）
# ============================================================

declare -A OCCUPIED_SERVER   # group_id -> host_alias
declare -A OCCUPIED_GPUS     # group_id -> gpu_ids_csv
declare -A OCCUPIED_SESSION  # group_id -> tmux_session_name
OCCUPIED_COUNT=0

release_all_gpus() {
    if (( OCCUPIED_COUNT == 0 )); then
        return 0
    fi

    echo ""
    echo "=== Releasing all occupied GPUs ==="
    for ((i = 1; i <= OCCUPIED_COUNT; i++)); do
        if [[ -n "${OCCUPIED_SERVER[$i]:-}" && -n "${OCCUPIED_SESSION[$i]:-}" ]]; then
            release_gpu_occupation "${OCCUPIED_SERVER[$i]}" "${OCCUPIED_SESSION[$i]}" || true
        fi
    done
    echo "All GPUs released."
}

cleanup() {
    echo ""
    echo "Received interrupt signal, cleaning up..."
    release_all_gpus
    exit 0
}

trap cleanup INT TERM

# ============================================================
# 获取目标服务器列表（按后缀数字降序）
# ============================================================

get_target_hosts() {
    local all_hosts
    all_hosts="$(get_hosts_from_ssh_config)"

    if [[ ${#TARGET_SERVERS[@]} -eq 0 ]]; then
        # 不过滤，返回所有服务器
        printf '%s\n' "$all_hosts" | sort -t_ -k3 -nr
        return
    fi

    # 按 TARGET_SERVERS 顺序过滤
    local host
    for target in "${TARGET_SERVERS[@]}"; do
        local normalized
        normalized="$(normalize_ssh_host "$target")"
        # 在所有主机中查找匹配
        while IFS= read -r host; do
            if [[ "$host" == "$normalized" ]]; then
                printf '%s\n' "$host"
                break
            fi
        done <<< "$all_hosts"
    done
}

# ============================================================
# 状态输出
# ============================================================

print_status_header() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "[$timestamp] === GPU Snatcher Status ==="
    echo "  Occupied: $OCCUPIED_COUNT / $NUM_GROUPS groups"
}

# ============================================================
# 主循环
# ============================================================

main() {
    local mem_pct
    mem_pct="$(python3 -c "print(int($GPU_MEMORY_TARGET_RATIO * 100))")"
    echo "=== GPU Snatcher ==="
    echo "  Target servers:   ${TARGET_SERVERS[*]:-(all zju_ servers)}"
    echo "  GPUs per group:   $GPUS_PER_GROUP"
    echo "  Total groups:     $NUM_GROUPS"
    echo "  Memory target:    ${mem_pct}%"
    echo "  Memory headroom:  ${GPU_MEMORY_HEADROOM_MIB} MiB"
    echo "  Poll interval:    ${POLL_INTERVAL}s"
    echo "  Tmux prefix:      $TMUX_SESSION_PREFIX"
    if (( DRY_RUN )); then
        echo "  *** DRY RUN MODE ***"
    fi
    echo ""

    # 解析目标服务器列表
    local target_hosts
    target_hosts="$(get_target_hosts)"
    if [[ -z "$target_hosts" ]]; then
        echo "ERROR: No target servers found. Check TARGET_SERVERS and SSH config." >&2
        exit 1
    fi

    local host_count
    host_count=$(echo "$target_hosts" | wc -l)
    echo "  Monitoring $host_count server(s):"
    while IFS= read -r host; do
        echo "    - $host"
    done <<< "$target_hosts"
    echo ""

    local all_occupied=0

    while true; do
        print_status_header

        # -------- 如果还没占满，扫描服务器找空闲 GPU --------
        if (( OCCUPIED_COUNT < NUM_GROUPS )); then
            echo "  Scanning for free GPUs..."

            local servers_checked=0
            local servers_ok=0
            local servers_down=0

            while IFS= read -r host; do
                [[ -z "$host" ]] && continue
                servers_checked=$((servers_checked + 1))

                # 查询该服务器 GPU 状态
                local gpu_output
                gpu_output="$(get_host_gpu_status "$host")" || true

                local host_state="DOWN"
                local host_note=""
                local free_gpu_ids=()
                local free_gpu_usage=()
                local total_gpus=0

                while IFS='|' read -r row_type field1 field2 field3 field4 field5 field6 field7; do
                    if [[ "$row_type" == "HOST" ]]; then
                        host_state="$field2"
                        host_note="$field3"
                    elif [[ "$row_type" == "GPU" ]]; then
                        total_gpus=$((total_gpus + 1))
                        local gpu_status="$field3"
                        local gpu_index="$field2"
                        local usage_pct="$field6"
                        if [[ "$gpu_status" == "FREE" ]]; then
                            free_gpu_ids+=("$gpu_index")
                            free_gpu_usage+=("$usage_pct")
                        fi
                    fi
                done <<< "$gpu_output"

                if [[ "$host_state" == "DOWN" ]]; then
                    servers_down=$((servers_down + 1))
                    echo "    $host: DOWN"
                    continue
                fi

                servers_ok=$((servers_ok + 1))
                local free_count="${#free_gpu_ids[@]}"

                # 构建空闲 GPU 描述
                local free_desc=""
                if (( free_count > 0 )); then
                    local id_list=""
                    for gid in "${free_gpu_ids[@]}"; do
                        [[ -n "$id_list" ]] && id_list+=","
                        id_list+="$gid"
                    done
                    free_desc="free: GPU$id_list"
                else
                    free_desc="no free GPUs"
                fi
                echo "    $host: $free_count/$total_gpus free ($free_desc)"

                # 检查是否有足够的空闲 GPU
                if (( free_count >= GPUS_PER_GROUP )); then
                    # 选择前 N 个空闲 GPU
                    local selected_ids=""
                    local selected_usage=""
                    for ((j = 0; j < GPUS_PER_GROUP; j++)); do
                        [[ -n "$selected_ids" ]] && selected_ids+=","
                        selected_ids+="${free_gpu_ids[$j]}"
                        [[ -n "$selected_usage" ]] && selected_usage+=", "
                        selected_usage+="GPU${free_gpu_ids[$j]}(${free_gpu_usage[$j]}%)"
                    done

                    OCCUPIED_COUNT=$((OCCUPIED_COUNT + 1))
                    local group_id=$OCCUPIED_COUNT
                    local session_name="${TMUX_SESSION_PREFIX}_${group_id}"

                    OCCUPIED_SERVER[$group_id]="$host"
                    OCCUPIED_GPUS[$group_id]="$selected_ids"
                    OCCUPIED_SESSION[$group_id]="$session_name"

                    echo ""
                    echo "  >>> Occupying group $group_id: $host GPU$selected_ids ($selected_usage)"
                    if launch_gpu_occupation "$host" "$session_name" "$selected_ids"; then
                        echo "  >>> Group $group_id launched successfully"
                    else
                        echo "  >>> ERROR: Failed to launch group $group_id on $host"
                        OCCUPIED_COUNT=$((OCCUPIED_COUNT - 1))
                        unset "OCCUPIED_SERVER[$group_id]"
                        unset "OCCUPIED_GPUS[$group_id]"
                        unset "OCCUPIED_SESSION[$group_id]"
                    fi

                    if (( OCCUPIED_COUNT >= NUM_GROUPS )); then
                        break
                    fi
                fi
            done <<< "$target_hosts"

            echo "  Servers: $servers_checked checked, $servers_ok OK, $servers_down DOWN"
        fi

        # -------- 打印已占组详情 --------
        if (( OCCUPIED_COUNT > 0 )); then
            echo ""
            echo "  --- Occupied Groups ---"
            for ((i = 1; i <= OCCUPIED_COUNT; i++)); do
                echo "  [$i] ${OCCUPIED_SERVER[$i]:-?} GPU${OCCUPIED_GPUS[$i]:-?} (tmux: ${OCCUPIED_SESSION[$i]:-?})"
            done
        fi

        # -------- 占满了，等待用户交互 --------
        if (( OCCUPIED_COUNT >= NUM_GROUPS )); then
            if (( ! all_occupied )); then
                all_occupied=1
                echo ""
                echo "=============================================="
                echo "  All $NUM_GROUPS group(s) occupied!"
                echo "  Press Enter to release all GPUs and exit..."
                echo "  (Ctrl+C also releases GPUs safely)"
                echo "=============================================="
                echo ""
            else
                echo ""
                echo "  Waiting... Press Enter to release all GPUs and exit."
            fi

            # 阻塞等待用户按 Enter
            read -r __unused
            release_all_gpus
            echo "Done. Exiting."
            exit 0
        fi

        # -------- 还没占满，等一轮再查 --------
        echo ""
        echo "  Sleeping ${POLL_INTERVAL}s before next check..."
        sleep "$POLL_INTERVAL"
    done
}

main "$@"
