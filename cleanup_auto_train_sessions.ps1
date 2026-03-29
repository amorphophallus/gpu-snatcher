param(
    [string]$SshConfigPath = "$HOME/.ssh/config",
    [int]$ConnectTimeoutSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sessionNameCandidates = @(
    "atlas",
    "birch",
    "cedar",
    "comet",
    "delta",
    "ember",
    "lotus",
    "maple",
    "nova",
    "pine",
    "river",
    "stone"
)

function Get-ZjuHostsFromSshConfig {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "SSH config not found: $Path"
    }

    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $result = [System.Collections.Generic.List[string]]::new()

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) {
            continue
        }

        if ($trimmed -match '^(?i)Host\s+(.+)$') {
            foreach ($pattern in ($Matches[1] -split '\s+')) {
                if ($pattern -like 'zju_4090_*' -and $pattern -notmatch '[*?]') {
                    if ($seen.Add($pattern)) {
                        $result.Add($pattern)
                    }
                }
            }
        }
    }

    return @($result)
}

function Invoke-SshCommand {
    param(
        [string]$HostAlias,
        [string]$RemoteCommand
    )

    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$ConnectTimeoutSeconds",
        $HostAlias,
        $RemoteCommand
    )

    $hasNativePreference = $null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)
    if ($hasNativePreference) {
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
        $output = & ssh @sshArgs 2>&1
    } finally {
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }

    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($output | Out-String).TrimEnd()
    }
}

$rows = foreach ($hostAlias in (Get-ZjuHostsFromSshConfig -Path $SshConfigPath)) {
    $killed = [System.Collections.Generic.List[string]]::new()
    $sessionList = ($sessionNameCandidates | ForEach-Object { "'$_'" }) -join ' '
    $remoteTemplate = @'
for s in {0}; do
    if tmux has-session -t "$s" 2>/dev/null; then
        tmux kill-session -t "$s" >/dev/null 2>&1 && printf '%s\n' "$s"
    fi
done
exit 0
'@
    $remoteCommand = [string]::Format($remoteTemplate, $sessionList)
    $result = Invoke-SshCommand -HostAlias $hostAlias -RemoteCommand $remoteCommand

    $reachable = $result.ExitCode -eq 0
    $note = if ($reachable) { '' } else { if ($result.Output) { $result.Output } else { "SSH failed with exit code $($result.ExitCode)" } }

    if ($reachable -and $result.Output) {
        foreach ($line in ($result.Output -split "`r?`n")) {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $killed.Add($line.Trim())
            }
        }
    }

    [pscustomobject]@{
        Host            = $hostAlias
        Reachable       = $reachable
        KilledSessions  = if ($killed.Count -gt 0) { $killed -join ', ' } else { '-' }
        Note            = if ($reachable) { '' } else { $note }
    }
}

$rows | Format-Table -AutoSize
