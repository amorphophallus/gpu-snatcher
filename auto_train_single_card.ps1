param(
    [string]$SshConfigPath = "$HOME/.ssh/config",
    [double]$MemoryUsageThreshold = 0.1,
    [int]$ConnectTimeoutSeconds = 5,
    [int]$PollIntervalSeconds = 5,
    [int]$PollTimeoutSeconds = 900,
    [string]$RemoteProjectDir = "/mnt/nas/share/home/hy/robust-rearrangement-custom",
    [string]$RemoteCondaEnv = "rr"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$global:TRAIN_COMMAND = "python -m src.train.bc +experiment=rgbd/diff_unet task='[one_leg,round_table,lamp]' data.demo_source=rollout data.demo_outcome=success data.suffix=rgbd-skill data.data_subset=50 training.num_epochs=4000 wandb.project=multi-task-rgbd-skill-low training.gpu_id=6 randomness=low dryrun=false wandb.continue_run_id=e56mvprj"
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

function Get-FirstFreeGpu {
    $candidates = @()

    foreach ($hostAlias in (Get-ZjuHostsFromSshConfig -Path $SshConfigPath)) {
        $result = Invoke-SshCommand -HostAlias $hostAlias -RemoteCommand `
            'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'

        if ($result.ExitCode -ne 0) {
            continue
        }

        foreach ($line in ($result.Output -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            $parts = $line -split '\s*,\s*'
            if ($parts.Count -lt 4) {
                continue
            }

            $gpuId = [int]$parts[0]
            $memoryTotal = [double]$parts[1]
            $memoryUsed = [double]$parts[2]
            $gpuUtil = [double]$parts[3]
            $usageRatio = if ($memoryTotal -gt 0) { $memoryUsed / $memoryTotal } else { 1.0 }

            if ($usageRatio -lt $MemoryUsageThreshold) {
                $candidates += [pscustomobject]@{
                    HostAlias = $hostAlias
                    GpuId     = $gpuId
                    GpuUtil   = $gpuUtil
                    MemoryUsed = $memoryUsed
                }
            }
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates | Sort-Object GpuUtil, MemoryUsed, HostAlias, GpuId | Select-Object -First 1
    }

    throw "No reachable free GPU found."
}

function Update-TrainCommandGpuId {
    param(
        [string]$Command,
        [int]$GpuId
    )

    if ([string]::IsNullOrWhiteSpace($Command)) {
        throw "Set TRAIN_COMMAND at the top of this script before running it."
    }

    if ($Command -match '(^|\s)training\.gpu_id=\S+') {
        return [regex]::Replace($Command, '(^|\s)training\.gpu_id=\S+', "`$1training.gpu_id=$GpuId", 1)
    }

    return "$Command training.gpu_id=$GpuId"
}

function Get-CommandName {
    param([string]$Command)

    if ([string]::IsNullOrWhiteSpace($Command)) {
        return "unknown"
    }

    $tokens = [regex]::Matches($Command, '(?:"(?:\\.|[^"])*"|''(?:\\.|[^''])*''|\S+)') | ForEach-Object Value
    $filtered = [System.Collections.Generic.List[string]]::new()

    foreach ($token in $tokens) {
        if ($token -match '^[A-Za-z_][A-Za-z0-9_]*=') {
            continue
        }
        $filtered.Add($token.Trim("'`""))
    }

    if ($filtered.Count -eq 0) {
        return "unknown"
    }

    $first = [System.IO.Path]::GetFileName($filtered[0])
    if ($first -match '^python(\d+(\.\d+)*)?(\.exe)?$' -and $filtered.Count -gt 1) {
        if ($filtered.Count -gt 2 -and $filtered[1] -eq '-m') {
            return $filtered[2]
        }

        if (-not $filtered[1].StartsWith('-')) {
            return [System.IO.Path]::GetFileName($filtered[1])
        }
    }

    return $first
}

function Get-AvailableTmuxSessionName {
    param([string]$HostAlias)

    foreach ($candidate in $sessionNameCandidates) {
        $result = Invoke-SshCommand -HostAlias $HostAlias -RemoteCommand "tmux has-session -t '$candidate' >/dev/null 2>&1"
        if ($result.ExitCode -ne 0) {
            return $candidate
        }
    }

    throw "No available tmux session name in candidate list."
}

function Start-RemoteTraining {
    param(
        [string]$HostAlias,
        [string]$SessionName,
        [string]$PreparedCommand
    )

    $remoteScript = @'
set -euo pipefail
session_name="$1"
project_dir="$2"
conda_env="$3"
encoded_train_command="$4"
train_command="$(printf '%s' "$encoded_train_command" | base64 -d)"

command -v tmux >/dev/null 2>&1
tmux has-session -t "$session_name" >/dev/null 2>&1 && exit 10
tmux new-session -d -s "$session_name"
tmux set-option -t "$session_name" remain-on-exit on
tmux new-window -t "$session_name" -n train
tmux send-keys -t "$session_name:train" -l "cd $project_dir"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "source ~/.bashrc >/dev/null 2>&1 || true"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l 'eval "$(conda shell.bash hook 2>/dev/null)" || true'
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "export TMPDIR=/tmp TEMP=/tmp TMP=/tmp"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "conda activate $conda_env"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "echo __AUTO_TRAIN_READY__"
tmux send-keys -t "$session_name:train" Enter
tmux send-keys -t "$session_name:train" -l "$train_command"
tmux send-keys -t "$session_name:train" Enter
tmux kill-window -t "${session_name}:0" >/dev/null 2>&1 || true
'@

    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$ConnectTimeoutSeconds",
        $HostAlias,
        'bash', '-s', '--',
        $SessionName,
        $RemoteProjectDir,
        $RemoteCondaEnv,
        ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($PreparedCommand)))
    )

    $output = $remoteScript | & ssh @sshArgs 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($output | Out-String).TrimEnd()
    }
}

function Get-TmuxPaneOutput {
    param(
        [string]$HostAlias,
        [string]$SessionName
    )

    return Invoke-SshCommand -HostAlias $HostAlias -RemoteCommand "tmux capture-pane -pt '$SessionName:train' -S -200"
}

function Get-WandbRunName {
    param([string]$Text)

    $cleanText = [regex]::Replace($Text, '\x1b\[[0-9;]*[A-Za-z]', '')
    foreach ($line in ($cleanText -split "`r?`n")) {
        if ($line -match '(?i)wandb run name\s*[:=]\s*(.+)$') {
            return $Matches[1].Trim()
        }
        if ($line -match '(?i)wandb[: ]+run name\s*[:=]\s*(.+)$') {
            return $Matches[1].Trim()
        }
    }

    return $null
}

function Get-TrainingFailureReason {
    param([string]$Text)

    $cleanText = [regex]::Replace($Text, '\x1b\[[0-9;]*[A-Za-z]', '')
    $patterns = @(
        '(?i)^Traceback \(most recent call last\):.*',
        '(?i)^.*Error executing job with overrides:.*',
        '(?i)^.*FileNotFoundError:.*',
        '(?i)^.*ModuleNotFoundError:.*',
        '(?i)^.*RuntimeError:.*',
        '(?i)^.*OSError:.*',
        '(?i)^.*AssertionError:.*',
        '(?i)^.*UnboundLocalError:.*',
        '(?i)^.*ValueError:.*',
        '(?i)^.*KeyError:.*',
        '(?i)^.*IndexError:.*',
        '(?i)^.*TypeError:.*',
        '(?i)^.*No space left on device.*',
        '(?i)^.*command not found.*',
        '(?i)^.*Killed$'
    )

    $failureLines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in ($cleanText -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed) {
            continue
        }

        foreach ($pattern in $patterns) {
            if ($trimmed -match $pattern) {
                if (-not $failureLines.Contains($trimmed)) {
                    [void]$failureLines.Add([string]$trimmed)
                }
                break
            }
        }
    }

    if ($failureLines.Count -gt 0) {
        return ($failureLines -join ' | ')
    }

    return $null
}

if ([string]::IsNullOrWhiteSpace($TRAIN_COMMAND)) {
    throw "Set TRAIN_COMMAND at the top of this script before running it."
}

$selection = Get-FirstFreeGpu
$preparedCommand = Update-TrainCommandGpuId -Command $TRAIN_COMMAND -GpuId $selection.GpuId
$commandName = Get-CommandName -Command $preparedCommand
$sessionName = Get-AvailableTmuxSessionName -HostAlias $selection.HostAlias
$startResult = Start-RemoteTraining -HostAlias $selection.HostAlias -SessionName $sessionName -PreparedCommand $preparedCommand

if ($startResult.ExitCode -ne 0) {
    throw "Failed to start tmux session '$sessionName' on $($selection.HostAlias): $($startResult.Output)"
}

$startTime = Get-Date
$wandbRunName = $null
$failureReason = $null
$lastPaneOutput = ""

while (-not $wandbRunName) {
    Start-Sleep -Seconds $PollIntervalSeconds

    $paneResult = Get-TmuxPaneOutput -HostAlias $selection.HostAlias -SessionName $sessionName
    if ($paneResult.ExitCode -ne 0) {
        throw "Failed to capture tmux output from $($selection.HostAlias):$sessionName : $($paneResult.Output)"
    }

    $lastPaneOutput = $paneResult.Output
    $wandbRunName = Get-WandbRunName -Text $lastPaneOutput
    $failureReason = Get-TrainingFailureReason -Text $lastPaneOutput

    if ($wandbRunName) {
        break
    }

    if ($failureReason) {
        Write-Host "status: failed"
        Write-Host "server: $($selection.HostAlias)"
        Write-Host "gpu_id: $($selection.GpuId)"
        Write-Host "gpu_util: $([math]::Round($selection.GpuUtil, 0))"
        Write-Host "tmux_name: $sessionName"
        Write-Host "command_name: $commandName"
        Write-Host "wandb_run_name: -"
        Write-Host "error_reason: $failureReason"
        exit 1
    }

    if (((Get-Date) - $startTime).TotalSeconds -ge $PollTimeoutSeconds) {
        Write-Host "status: timeout"
        Write-Host "server: $($selection.HostAlias)"
        Write-Host "gpu_id: $($selection.GpuId)"
        Write-Host "gpu_util: $([math]::Round($selection.GpuUtil, 0))"
        Write-Host "tmux_name: $sessionName"
        Write-Host "command_name: $commandName"
        Write-Host "wandb_run_name: -"
        Write-Host "error_reason: Timed out waiting for wandb run name"
        exit 1
    }
}

Write-Host "status: started"
Write-Host "server: $($selection.HostAlias)"
Write-Host "gpu_id: $($selection.GpuId)"
Write-Host "gpu_util: $([math]::Round($selection.GpuUtil, 0))"
Write-Host "tmux_name: $sessionName"
Write-Host "command_name: $commandName"
Write-Host "wandb_run_name: $wandbRunName"
