param(
    [string]$SshConfigPath = "$HOME/.ssh/config",
    [double]$MemoryUsageThreshold = 0.1,
    [int]$ConnectTimeoutSeconds = 5,
    [int]$PollIntervalSeconds = 5,
    [int]$PollTimeoutSeconds = 300,
    [int]$SshCommandTimeoutSeconds = 15,
    [string]$RemoteProjectDir = "/mnt/nas/share/home/hy/robust-rearrangement-custom",
    [string]$RemoteCondaEnv = "rr"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$global:TRAIN_COMMAND_PARTS = @(
    "python",
    "-m",
    "src.train.bc",
    "+experiment=rgbd/diff_unet",
    "task=[one_leg,round_table,lamp]",
    "data.demo_source=rollout",
    "data.demo_outcome=success",
    "data.suffix=rgbd-skill",
    "data.data_subset=50",
    "training.batch_size=256",
    "training.num_epochs=4000",
    "training.steps_per_epoch=-1",
    "training.save_per_epoch=1000",
    "wandb.project=multi-task-rgbd-skill-low",
    "training.gpu_id=7",
    "randomness=low",
    "dryrun=false",
    "wandb.continue_run_id=e56mvprj"
)
$global:TRAIN_COMMAND = [string]::Join(' ', ($global:TRAIN_COMMAND_PARTS | ForEach-Object {
    if ($_ -match '[\s"]') {
        '"' + ($_ -replace '"', '\"') + '"'
    } else {
        $_
    }
}))
$global:SSH_NAME = "230"
$global:GPU_ID = "0"
$global:FAST_SERVER = @("236", "230")
$global:SLOW_SERVER = @("228", "238", "240")
$global:DATA_DIR_PROCESSED = ""
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
        [string]$RemoteCommand,
        [int]$CommandTimeoutSeconds = $SshCommandTimeoutSeconds
    )

    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$ConnectTimeoutSeconds",
        '-o', 'ServerAliveInterval=5',
        '-o', 'ServerAliveCountMax=1',
        $HostAlias,
        $RemoteCommand
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "ssh"
    $psi.Arguments = ($sshArgs | ForEach-Object {
        if ($_ -match '\s|"') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join ' '
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    [void]$process.Start()

    if (-not $process.WaitForExit($CommandTimeoutSeconds * 1000)) {
        try { $process.Kill() } catch {}
        return [pscustomobject]@{
            ExitCode = 124
            Output   = "SSH command timed out after $CommandTimeoutSeconds seconds."
        }
    }

    $stdOut = $process.StandardOutput.ReadToEnd()
    $stdErr = $process.StandardError.ReadToEnd()
    $output = ($stdOut + $stdErr).TrimEnd()

    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output   = $output
    }
}

function Invoke-GpuQueryCommand {
    param([string]$HostAlias)

    $query = 'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$ConnectTimeoutSeconds",
        '-o', 'StrictHostKeyChecking=accept-new',
        $HostAlias,
        $query
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
        Output   = ($output | Out-String).Trim()
    }
}

function Get-HostGpuStatus {
    param(
        [string]$HostAlias,
        [double]$Threshold
    )

    $result = Invoke-GpuQueryCommand -HostAlias $HostAlias

    if ($result.ExitCode -ne 0) {
        return [pscustomobject]@{
            Host              = $HostAlias
            Reachable         = $false
            Error             = if ($result.Output) { $result.Output } else { "SSH failed with exit code $($result.ExitCode)" }
            Gpus              = @()
            AvailableGpuCount = 0
            TotalGpuCount     = 0
        }
    }

    $gpus = @()
    foreach ($line in ($result.Output -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $parts = $line -split '\s*,\s*'
        if ($parts.Count -lt 4) {
            continue
        }

        $index = [int]$parts[0]
        $memoryTotal = [double]$parts[1]
        $memoryUsed = [double]$parts[2]
        $gpuUtil = [double]$parts[3]
        $usageRatio = if ($memoryTotal -gt 0) { $memoryUsed / $memoryTotal } else { 1.0 }
        $available = $usageRatio -lt $Threshold

        $gpus += [pscustomobject]@{
            Index        = $index
            MemoryTotal  = [math]::Round($memoryTotal, 0)
            MemoryUsed   = [math]::Round($memoryUsed, 0)
            UsagePercent = [math]::Round($usageRatio * 100, 1)
            GpuUtil      = [math]::Round($gpuUtil, 0)
            Available    = $available
            Status       = if ($available) { "FREE" } else { "BUSY" }
        }
    }

    $sortedGpus = @($gpus | Sort-Object Index)
    $availableGpuCount = @($sortedGpus | Where-Object Available).Count

    return [pscustomobject]@{
        Host              = $HostAlias
        Reachable         = $true
        Error             = $null
        Gpus              = $sortedGpus
        AvailableGpuCount = $availableGpuCount
        TotalGpuCount     = @($sortedGpus).Count
    }
}

function Write-StructuredStatusAndExit {
    param(
        [string]$Status,
        [string]$Server,
        [int]$GpuId,
        [double]$GpuUtil,
        [string]$TmuxName,
        [string]$CommandName,
        [string]$WandbRunName,
        [string]$ErrorReason,
        [int]$ExitCode
    )

    Write-Host "status: $Status"
    Write-Host "server: $Server"
    Write-Host "gpu_id: $GpuId"
    Write-Host "gpu_util: $([math]::Round($GpuUtil, 0))"
    Write-Host "tmux_name: $TmuxName"
    Write-Host "command_name: $CommandName"
    Write-Host "wandb_run_name: $WandbRunName"
    Write-Host "error_reason: $ErrorReason"
    exit $ExitCode
}

function Get-FirstFreeGpu {
    $candidates = @()

    foreach ($hostAlias in (Get-ZjuHostsFromSshConfig -Path $SshConfigPath)) {
        $status = Get-HostGpuStatus -HostAlias $hostAlias -Threshold $MemoryUsageThreshold
        if (-not $status.Reachable) {
            continue
        }

        foreach ($gpu in ($status.Gpus | Where-Object Available)) {
                $candidates += [pscustomobject]@{
                    HostAlias = $hostAlias
                    GpuId     = $gpu.Index
                    GpuUtil   = $gpu.GpuUtil
                    MemoryUsed = $gpu.MemoryUsed
                }
        }
    }

    $fastSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($serverId in $FAST_SERVER) {
        if (-not [string]::IsNullOrWhiteSpace($serverId)) {
            [void]$fastSet.Add(([string]$serverId).Trim())
        }
    }

    $slowSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($serverId in $SLOW_SERVER) {
        if (-not [string]::IsNullOrWhiteSpace($serverId)) {
            [void]$slowSet.Add(([string]$serverId).Trim())
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates |
            Sort-Object `
                GpuUtil, `
                MemoryUsed, `
                @{ Expression = {
                        $suffix = if ($_.HostAlias -match '(\d+)$') { $Matches[1] } else { '' }
                        if ($suffix -and $fastSet.Contains($suffix)) {
                            0
                        } elseif ($suffix -and $slowSet.Contains($suffix)) {
                            2
                        } else {
                            1
                        }
                    } }, `
                @{ Expression = {
                        if ($_.HostAlias -match '(\d+)$') {
                            [int]$Matches[1]
                        } else {
                            -1
                        }
                    }; Descending = $true }, `
                HostAlias, `
                GpuId |
            Select-Object -First 1
    }

    throw "No reachable free GPU found."
}

function Get-PreferredGpuOrThrow {
    param(
        [string]$SshName,
        [string]$GpuIdText
    )

    if (-not ($GpuIdText -match '^\d+$')) {
        throw "GPU_ID must be a non-negative integer, got '$GpuIdText'."
    }

    $hostAlias = "zju_4090_$SshName"
    $targetGpuId = [int]$GpuIdText

    $status = Get-HostGpuStatus -HostAlias $hostAlias -Threshold $MemoryUsageThreshold
    if (-not $status.Reachable) {
        throw "Preferred host '$hostAlias' is unreachable: $($status.Error)"
    }

    $gpu = @($status.Gpus | Where-Object { $_.Index -eq $targetGpuId } | Select-Object -First 1)
    if ($gpu.Count -eq 0) {
        $availableGpuIds = if ($status.Gpus.Count -gt 0) { ($status.Gpus | ForEach-Object { $_.Index.ToString() }) -join ', ' } else { 'none' }
        throw "Preferred GPU not found on host '$hostAlias': GPU$targetGpuId. Available GPU IDs reported by nvidia-smi: $availableGpuIds. GPU IDs are 0-based."
    }

    if (-not $gpu[0].Available) {
        throw "Preferred GPU not available: $hostAlias GPU$targetGpuId memory usage $($gpu[0].UsagePercent)% >= threshold $([math]::Round($MemoryUsageThreshold * 100, 1))% (gpu util $($gpu[0].GpuUtil)%)." 
    }

    return [pscustomobject]@{
        HostAlias = $hostAlias
        GpuId     = $gpu[0].Index
        GpuUtil   = $gpu[0].GpuUtil
        MemoryUsed = $gpu[0].MemoryUsed
    }
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
        [regex]::Replace($Command, '(^|\s)training\.gpu_id=\S+', "`$1training.gpu_id=$GpuId", 1)
    } else {
        "$Command training.gpu_id=$GpuId"
    }
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
        [string]$PreparedCommand,
        [string]$DataDirProcessed
    )

    $remoteScript = @'
set -euo pipefail
session_name="$1"
project_dir="$2"
conda_env="$3"
encoded_train_command="$4"
data_dir_processed="${5:-}"
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
if [[ -n "${data_dir_processed// }" ]]; then
    printf -v data_dir_export 'export DATA_DIR_PROCESSED=%q' "$data_dir_processed"
    tmux send-keys -t "$session_name:train" -l "$data_dir_export"
    tmux send-keys -t "$session_name:train" Enter
fi
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
        ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($PreparedCommand))),
        $DataDirProcessed
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

$selection = if (-not [string]::IsNullOrWhiteSpace($SSH_NAME) -and -not [string]::IsNullOrWhiteSpace($GPU_ID)) {
    Get-PreferredGpuOrThrow -SshName $SSH_NAME -GpuIdText $GPU_ID
} else {
    Get-FirstFreeGpu
}
$preparedCommand = Update-TrainCommandGpuId -Command $TRAIN_COMMAND -GpuId $selection.GpuId
$commandName = Get-CommandName -Command $preparedCommand
$sessionName = Get-AvailableTmuxSessionName -HostAlias $selection.HostAlias
$startResult = Start-RemoteTraining -HostAlias $selection.HostAlias -SessionName $sessionName -PreparedCommand $preparedCommand -DataDirProcessed $DATA_DIR_PROCESSED

if ($startResult.ExitCode -ne 0) {
    throw "Failed to start tmux session '$sessionName' on $($selection.HostAlias): $($startResult.Output)"
}

$startTime = Get-Date
$wandbRunName = $null
$failureReason = $null
$lastPaneOutput = ""

while (-not $wandbRunName) {
    $elapsedSeconds = ((Get-Date) - $startTime).TotalSeconds
    if ($elapsedSeconds -ge $PollTimeoutSeconds) {
        Write-StructuredStatusAndExit `
            -Status "timeout" `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName "-" `
            -ErrorReason "Timed out waiting for wandb run name" `
            -ExitCode 1
    }

    Start-Sleep -Seconds $PollIntervalSeconds

    $elapsedSeconds = ((Get-Date) - $startTime).TotalSeconds
    if ($elapsedSeconds -ge $PollTimeoutSeconds) {
        Write-StructuredStatusAndExit `
            -Status "timeout" `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName "-" `
            -ErrorReason "Timed out waiting for wandb run name" `
            -ExitCode 1
    }

    $remainingSeconds = [math]::Ceiling($PollTimeoutSeconds - $elapsedSeconds)
    $commandTimeoutSeconds = [math]::Max(1, [math]::Min($SshCommandTimeoutSeconds, $remainingSeconds))
    $paneResult = Invoke-SshCommand -HostAlias $selection.HostAlias -RemoteCommand "tmux capture-pane -pt '$sessionName:train' -S -200" -CommandTimeoutSeconds $commandTimeoutSeconds
    if ($paneResult.ExitCode -ne 0) {
        $elapsedSeconds = ((Get-Date) - $startTime).TotalSeconds
        if ($paneResult.ExitCode -eq 124 -or $elapsedSeconds -ge $PollTimeoutSeconds) {
            Write-StructuredStatusAndExit `
                -Status "timeout" `
                -Server $selection.HostAlias `
                -GpuId $selection.GpuId `
                -GpuUtil $selection.GpuUtil `
                -TmuxName $sessionName `
                -CommandName $commandName `
                -WandbRunName "-" `
                -ErrorReason "Timed out while waiting for tmux output" `
                -ExitCode 1
        }

        throw "Failed to capture tmux output from $($selection.HostAlias):$sessionName : $($paneResult.Output)"
    }

    $lastPaneOutput = $paneResult.Output
    $wandbRunName = Get-WandbRunName -Text $lastPaneOutput
    $failureReason = Get-TrainingFailureReason -Text $lastPaneOutput

    if ($wandbRunName) {
        break
    }

    if ($failureReason) {
        Write-StructuredStatusAndExit `
            -Status "failed" `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName "-" `
            -ErrorReason $failureReason `
            -ExitCode 1
    }
}

Write-Host "status: started"
Write-Host "server: $($selection.HostAlias)"
Write-Host "gpu_id: $($selection.GpuId)"
Write-Host "gpu_util: $([math]::Round($selection.GpuUtil, 0))"
Write-Host "tmux_name: $sessionName"
Write-Host "command_name: $commandName"
Write-Host "wandb_run_name: $wandbRunName"
