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

    $output = & ssh @sshArgs 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($output | Out-String).TrimEnd()
    }
}

$rows = foreach ($hostAlias in (Get-ZjuHostsFromSshConfig -Path $SshConfigPath)) {
    $killed = [System.Collections.Generic.List[string]]::new()
    $reachable = $true
    $note = ""

    foreach ($sessionName in $sessionNameCandidates) {
        $result = Invoke-SshCommand -HostAlias $hostAlias -RemoteCommand "tmux kill-session -t '$sessionName' >/dev/null 2>&1"
        if ($result.ExitCode -eq 0) {
            $killed.Add($sessionName)
            continue
        }

        if ($result.Output -and $result.Output -notmatch 'can.t find session') {
            $reachable = $false
            $note = $result.Output
            break
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
