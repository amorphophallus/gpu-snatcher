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
                if ($i ~ /^zju_/ && $i !~ /[*?]/) {
                    print $i
                }
            }
        }
    ' "$SSH_CONFIG_PATH" | awk '!seen[$0]++'
}

invoke_ssh() {
    local host_alias="$1"
    local remote_command="$2"

    ssh -n \
        -o BatchMode=yes \
        -o ConnectTimeout="$CONNECT_TIMEOUT_SECONDS" \
        "$host_alias" \
        "$remote_command"
}

printf '%-16s %-10s %-22s %s\n' "Host" "Reachable" "KilledSessions" "RunningSessions(procs)"

session_list=""
for session_name in "${SESSION_NAME_CANDIDATES[@]}"; do
    session_list+=" '$session_name'"
done

while IFS= read -r host_alias; do
    [[ -z "$host_alias" ]] && continue

    remote_command="
for s in$session_list; do
    if tmux has-session -t \"\$s\" 2>/dev/null; then
        _running=''
        for _ppid in \$(tmux list-panes -t \"\$s\" -F '#{pane_pid}' 2>/dev/null); do
            [ -z \"\$_ppid\" ] && continue
            for _cpid in \$(ps --ppid \"\$_ppid\" -o pid= 2>/dev/null); do
                [ -z \"\$_cpid\" ] && continue
                _ccomm=\"\$(ps -o comm= -p \"\$_cpid\" 2>/dev/null)\"
                case \"\$_ccomm\" in
                    python|python3|torchrun)
                        _running=\"\${_running},\$_ccomm\"
                        continue
                        ;;
                esac
                for _gcid in \$(ps --ppid \"\$_cpid\" -o pid= 2>/dev/null); do
                    [ -z \"\$_gcid\" ] && continue
                    _gcomm=\"\$(ps -o comm= -p \"\$_gcid\" 2>/dev/null)\"
                    case \"\$_gcomm\" in
                        python|python3|torchrun)
                            _running=\"\${_running},\$_gcomm\"
                            ;;
                    esac
                done
            done
        done
        if [ -n \"\$_running\" ]; then
            _running=\"\$(printf '%s' \"\$_running\" | sed 's/^,//' | tr ',' '\\n' | sort -u | tr '\\n' ',' | sed 's/,\$//')\"
            printf 'RUNNING|%s|%s\\n' \"\$s\" \"\$_running\"
        else
            tmux kill-session -t \"\$s\" >/dev/null 2>&1 && printf 'KILLED|%s\\n' \"\$s\"
        fi
    fi
done
exit 0
"

    ssh_tmpout="$(mktemp)"
    ssh_rc=0
    invoke_ssh "$host_alias" "$remote_command" >"$ssh_tmpout" 2>&1 || ssh_rc=$?
    output="$(cat "$ssh_tmpout")"
    rm -f "$ssh_tmpout"

    if [[ $ssh_rc -eq 0 ]]; then
        killed=()
        running=()
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            case "$line" in
                KILLED\|*)
                    killed+=("${line#KILLED|}")
                    ;;
                RUNNING\|*)
                    running_info="${line#RUNNING|}"
                    running_name="${running_info%%|*}"
                    running_procs="${running_info#*|}"
                    running+=("${running_name}(${running_procs})")
                    ;;
            esac
        done <<< "$output"

        killed_str="-"
        running_str="-"
        if [[ ${#killed[@]} -gt 0 ]]; then
            killed_str="$(IFS=', '; echo "${killed[*]}")"
        fi
        if [[ ${#running[@]} -gt 0 ]]; then
            running_str="$(IFS=', '; echo "${running[*]}")"
        fi
        printf '%-16s %-10s %-22s %s\n' "$host_alias" "yes" "$killed_str" "$running_str"
    else
        first_line="$(printf '%s\n' "$output" | head -n1)"
        printf '%-16s %-10s %-22s %s\n' "$host_alias" "no" "${first_line:-ssh error}" "-"
    fi
done < <(get_hosts_from_ssh_config)
