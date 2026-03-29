#!/usr/bin/env bash

set -euo pipefail

SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-5}"

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

printf '%-16s %-10s %s\n' "Host" "Reachable" "KilledSessions"
while IFS= read -r host_alias; do
    [[ -z "$host_alias" ]] && continue

    session_list=""
    for session_name in "${SESSION_NAME_CANDIDATES[@]}"; do
        session_list+=" '$session_name'"
    done

    remote_command="for s in$session_list; do if tmux has-session -t \"\$s\" 2>/dev/null; then tmux kill-session -t \"\$s\" >/dev/null 2>&1 && printf '%s\\n' \"\$s\"; fi; done; exit 0"

    if output="$(invoke_ssh "$host_alias" "$remote_command" 2>&1)"; then
        reachable="yes"
        mapfile -t killed < <(printf '%s\n' "$output" | sed '/^[[:space:]]*$/d')
    else
        reachable="no"
        killed=("$output")
    fi

    if [[ "$reachable" == "yes" ]]; then
        if [[ ${#killed[@]} -eq 0 ]]; then
            printf '%-16s %-10s %s\n' "$host_alias" "$reachable" "-"
        else
            printf '%-16s %-10s %s\n' "$host_alias" "$reachable" "$(IFS=', '; echo "${killed[*]}")"
        fi
    else
        printf '%-16s %-10s %s\n' "$host_alias" "$reachable" "${killed[0]}"
    fi
done < <(get_hosts_from_ssh_config)
