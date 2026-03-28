#!/usr/bin/env bash

set -uo pipefail

SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
MEMORY_USAGE_THRESHOLD="${MEMORY_USAGE_THRESHOLD:-0.1}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-5}"

get_hosts_from_ssh_config() {
    if [[ ! -f "$SSH_CONFIG_PATH" ]]; then
        echo "SSH config not found: $SSH_CONFIG_PATH" >&2
        return 1
    fi

    awk '
        BEGIN { IGNORECASE = 1 }
        /^[[:space:]]*Host[[:space:]]+/ {
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^zju_4090_/ && $i !~ /[*?]/) {
                    print $i
                }
            }
        }
    ' "$SSH_CONFIG_PATH" | sort -u
}

check_host() {
    local host_alias="$1"
    local query_output
    local ssh_status

    query_output="$(ssh -o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
        -o StrictHostKeyChecking=accept-new \
        "$host_alias" \
        "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>&1)"
    ssh_status=$?

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

main() {
    mapfile -t hosts < <(get_hosts_from_ssh_config)
    if [[ ${#hosts[@]} -eq 0 ]]; then
        echo "No explicit zju_4090_* hosts found in $SSH_CONFIG_PATH" >&2
        return 1
    fi

    declare -A host_state
    declare -A host_note
    declare -A host_free_ids
    declare -A host_free_count
    declare -A host_total_count
    declare -A gpu_rows

    local host
    local line
    for host in "${hosts[@]}"; do
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue

            IFS='|' read -r row_type row_host field1 field2 field3 field4 field5 <<< "$line"
            if [[ "$row_type" == "HOST" ]]; then
                host_state["$row_host"]="$field1"
                host_note["$row_host"]="$field2"
                host_free_count["$row_host"]=0
                host_total_count["$row_host"]=0
                host_free_ids["$row_host"]=""
                gpu_rows["$row_host"]=""
            elif [[ "$row_type" == "GPU" ]]; then
                local gpu_index="$field1"
                local gpu_status="$field2"
                local used_mib="$field3"
                local total_mib="$field4"
                local usage_percent="$field5"
                local gpu_util="$field6"
                local gpu_row

                host_total_count["$row_host"]=$(( ${host_total_count["$row_host"]} + 1 ))
                printf -v gpu_row '%-4s %-6s %-10s %-10s %-10s %-8s\n' \
                    "$gpu_index" "$gpu_status" "$used_mib" "$total_mib" "$usage_percent" "$gpu_util"
                gpu_rows["$row_host"]+="$gpu_row"

                if [[ "$gpu_status" == "FREE" ]]; then
                    host_free_count["$row_host"]=$(( ${host_free_count["$row_host"]} + 1 ))
                    if [[ -n "${host_free_ids["$row_host"]}" ]]; then
                        host_free_ids["$row_host"]+=", "
                    fi
                    host_free_ids["$row_host"]+="GPU${gpu_index}(${gpu_util}%)"
                fi
            fi
        done < <(check_host "$host")
    done

    echo
    echo "=== ZJU 4090 GPU Summary ==="
    printf '%-16s %-6s %-12s %-16s %s\n' "Host" "SSH" "GPUs" "FreeGPUIds" "Note"
    for host in "${hosts[@]}"; do
        local note=""
        local gpu_summary="-"
        local free_ids="-"
        if [[ "${host_state["$host"]:-DOWN}" == "OK" ]]; then
            gpu_summary="${host_free_count["$host"]}/${host_total_count["$host"]} free"
            if [[ -n "${host_free_ids["$host"]}" ]]; then
                free_ids="${host_free_ids["$host"]}"
            fi
            if [[ ${host_total_count["$host"]} -ne 8 ]]; then
                note="Expected 8 GPUs, got ${host_total_count["$host"]}"
            fi
        else
            note="${host_note["$host"]}"
        fi

        printf '%-16s %-6s %-12s %-16s %s\n' \
            "$host" \
            "${host_state["$host"]:-DOWN}" \
            "$gpu_summary" \
            "$free_ids" \
            "$note"
    done

    for host in "${hosts[@]}"; do
        echo
        if [[ "${host_state["$host"]:-DOWN}" != "OK" ]]; then
            echo "[$host] SSH unreachable"
            echo "  ${host_note["$host"]}"
            continue
        fi

        echo "[$host] ${host_free_count["$host"]}/${host_total_count["$host"]} GPUs free"
        printf '%-4s %-6s %-10s %-10s %-10s %-8s\n' "GPU" "Status" "Used(MiB)" "Total(MiB)" "MemUsage%" "GpuUtil%"
        printf '%s' "${gpu_rows["$host"]}"
    done

    echo
    echo "=== Recommended Targets ==="
    local found_usable=0
    for host in "${hosts[@]}"; do
        if [[ "${host_state["$host"]:-DOWN}" == "OK" && ${host_free_count["$host"]:-0} -gt 0 ]]; then
            echo "$host: GPU ${host_free_ids["$host"]}"
            found_usable=1
        fi
    done

    if [[ $found_usable -eq 0 ]]; then
        echo "No currently usable GPUs found."
    fi
}

main "$@"
