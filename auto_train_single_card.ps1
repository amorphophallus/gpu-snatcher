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

$global:DATA_STORAGE_FORMAT = "lmdb"
$global:DATA_LOAD_INTO_MEMORY = "false"
$global:DATA_PATHS_OVERRIDE = ""

# 单卡训练命令
$global:TRAIN_COMMAND_PARTS = @(
    "python",
    "-m",
    "src.train.bc",
    "+experiment=rgbd/diff_unet",
    "task=[one_leg,round_table,lamp]",
    "data.demo_source=rollout",
    "data.demo_outcome=success",
    "data.suffix=rgbd-skill",
    "data.storage_format=$global:DATA_STORAGE_FORMAT",
    "data.load_into_memory=$global:DATA_LOAD_INTO_MEMORY",
    "data.data_subset=50",
    "training.batch_size=256",
    "training.num_epochs=4000",
    "training.steps_per_epoch=-1",
    "training.save_per_epoch=1000",
    "wandb.project=multi-task-rgbd-skill-low",
    "wandb.mode=online",
    "training.gpu_id=7",
    "randomness=low",
    "dryrun=false",
    "wandb.continue_run_id=e56mvprj"
)
if (-not [string]::IsNullOrWhiteSpace($global:DATA_PATHS_OVERRIDE)) {
    $global:TRAIN_COMMAND_PARTS += "data.data_paths_override=$global:DATA_PATHS_OVERRIDE"
}

# 多卡训练命令
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

function ConvertTo-SshArgumentString {
    param([string[]]$Arguments)

    $quoted = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            '"' + ($argument -replace '"', '\"') + '"'
        } else {
            $argument
        }
    }

    return ($quoted -join ' ')
}

function Get-ZjuHostsFromSshConfig {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "SSH config not found: $Path"
    }

    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $result = [System.Collections.Generic.List[string]]::new()

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) {
            continue
        }

        if ($trimmed -match '^(?i)Host\s+(.+)$') {
            foreach ($pattern in ($Matches[1] -split '\s+')) {
                if ($pattern -match '^zju_4090_' -and $pattern -notmatch '[*?]') {
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
    $psi.FileName = 'ssh'
    $psi.Arguments = ConvertTo-SshArgumentString -Arguments $sshArgs
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    try {
        [void]$process.Start()
        if (-not $process.WaitForExit($CommandTimeoutSeconds * 1000)) {
            try { $process.Kill() } catch {}
            return [pscustomobject]@{
                ExitCode = 124
                Output   = "SSH command timed out after $CommandTimeoutSeconds seconds."
            }
        }

        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $combined = @($stdout, $stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output   = ($combined -join "`n").Trim()
        }
    } catch {
        return [pscustomobject]@{
            ExitCode = 255
            Output   = $_.Exception.Message
        }
    } finally {
        $process.Dispose()
    }
}

function Find-FirstFreeGpu {
    $candidates = [System.Collections.Generic.List[object]]::new()
    $fastSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $slowSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    foreach ($serverId in $FAST_SERVER) {
        if (-not [string]::IsNullOrWhiteSpace($serverId)) {
            [void]$fastSet.Add(([string]$serverId).Trim())
        }
    }

    foreach ($serverId in $SLOW_SERVER) {
        if (-not [string]::IsNullOrWhiteSpace($serverId)) {
            [void]$slowSet.Add(([string]$serverId).Trim())
        }
    }

    foreach ($hostAlias in (Get-ZjuHostsFromSshConfig -Path $SshConfigPath)) {
        if ([string]::IsNullOrWhiteSpace($hostAlias)) {
            continue
        }

        $result = Invoke-SshCommand -HostAlias $hostAlias -RemoteCommand 'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
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
                $suffix = ''
                $hostNum = 0
                if ($hostAlias -match '([0-9]+)$') {
                    $suffix = $Matches[1]
                    $hostNum = [int]$suffix
                }

                $priority = 1
                if ($suffix -and $fastSet.Contains($suffix)) {
                    $priority = 0
                } elseif ($suffix -and $slowSet.Contains($suffix)) {
                    $priority = 2
                }

                $candidates.Add([pscustomobject]@{
                    HostAlias  = $hostAlias
                    GpuId      = $gpuId
                    GpuUtil    = $gpuUtil
                    MemoryUsed = $memoryUsed
                    Priority   = $priority
                    HostNum    = $hostNum
                })
            }
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates |
            Sort-Object `
                @{ Expression = { $_.GpuUtil } }, `
                @{ Expression = { $_.MemoryUsed } }, `
                @{ Expression = { $_.Priority } }, `
                @{ Expression = { $_.HostNum }; Descending = $true }, `
                @{ Expression = { $_.HostAlias } }, `
                @{ Expression = { $_.GpuId } } |
            Select-Object -First 1
    }

    return $null
}

function Find-PreferredGpuOrError {
    param(
        [string]$SshNameValue,
        [string]$GpuIdText
    )

    if ($GpuIdText -notmatch '^\d+$') {
        [Console]::Error.WriteLine("GPU_ID must be a non-negative integer, got '$GpuIdText'.")
        return $null
    }

    $hostAlias = "zju_4090_$SshNameValue"
    $targetGpuId = [int]$GpuIdText
    $result = Invoke-SshCommand -HostAlias $hostAlias -RemoteCommand 'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'

    if ($result.ExitCode -ne 0) {
        [Console]::Error.WriteLine("Preferred host '$hostAlias' is unreachable: $($result.Output)")
        return $null
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
        if ($gpuId -ne $targetGpuId) {
            continue
        }

        $memoryTotal = [double]$parts[1]
        $memoryUsed = [double]$parts[2]
        $gpuUtil = [double]$parts[3]
        $usageRatio = if ($memoryTotal -gt 0) { $memoryUsed / $memoryTotal } else { 1.0 }
        $usagePercent = [math]::Round($usageRatio * 100, 1)
        $thresholdPercent = [math]::Round($MemoryUsageThreshold * 100, 1)

        if ($usageRatio -ge $MemoryUsageThreshold) {
            [Console]::Error.WriteLine("Preferred GPU not available: $hostAlias GPU$targetGpuId memory usage $usagePercent% >= threshold $thresholdPercent% (gpu util $([math]::Round($gpuUtil, 0))%).")
            return $null
        }

        return [pscustomobject]@{
            HostAlias  = $hostAlias
            GpuId      = $gpuId
            GpuUtil    = $gpuUtil
            MemoryUsed = $memoryUsed
        }
    }

    [Console]::Error.WriteLine("Preferred GPU not found on host '$hostAlias': GPU$targetGpuId.")
    return $null
}

function Prepare-TrainCommand {
    param([int]$GpuId)

    $command = $TRAIN_COMMAND.Trim()
    if (-not $command) {
        throw "TRAIN_COMMAND is empty."
    }

    if ($command -match '(^|\s)training\.gpu_id=\S+') {
        return [regex]::Replace($command, '(^|\s)training\.gpu_id=\S+', "`$1training.gpu_id=$GpuId", 1)
    }

    return "$command training.gpu_id=$GpuId"
}

function Get-CommandName {
    param([string]$Command)

    if ([string]::IsNullOrWhiteSpace($Command)) {
        return 'unknown'
    }

    $tokens = [regex]::Matches($Command.Trim(), '(?:"(?:\\.|[^"])*"|''(?:\\.|[^''])*''|\S+)') | ForEach-Object Value
    $parts = [System.Collections.Generic.List[string]]::new()

    foreach ($token in $tokens) {
        $clean = $token.Trim("'`"")
        $parts.Add($clean)
    }

    while ($parts.Count -gt 0 -and $parts[0].Contains('=') -and -not $parts[0].StartsWith('/') -and -not $parts[0].StartsWith('./')) {
        $key = $parts[0].Split('=', 2)[0]
        if ($key -notmatch '^[A-Za-z0-9_]+$') {
            break
        }
        $parts.RemoveAt(0)
    }

    if ($parts.Count -eq 0) {
        return 'unknown'
    }

    $first = [System.IO.Path]::GetFileName($parts[0])
    if ($first.StartsWith('python') -and $parts.Count -gt 1) {
        if ($parts.Count -gt 2 -and $parts[1] -eq '-m') {
            return $parts[2]
        }
        if (-not $parts[1].StartsWith('-')) {
            return [System.IO.Path]::GetFileName($parts[1])
        }
        return [System.IO.Path]::GetFileName($parts[0])
    }

    return [System.IO.Path]::GetFileName($parts[0])
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
        ([Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($PreparedCommand))),
        $DataDirProcessed
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = 'ssh'
    $psi.Arguments = ConvertTo-SshArgumentString -Arguments $sshArgs
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    try {
        [void]$process.Start()
        $process.StandardInput.Write($remoteScript)
        $process.StandardInput.Close()

        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()

        $combined = @($stdout, $stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output   = ($combined -join "`n").Trim()
        }
    } finally {
        $process.Dispose()
    }
}

function Capture-TmuxOutput {
    param(
        [string]$HostAlias,
        [string]$SessionName
    )

    return Invoke-SshCommand -HostAlias $HostAlias -RemoteCommand "tmux capture-pane -pt '$SessionName:train' -S -200"
}

function Get-WandbRunName {
    param([string]$Text)

    $cleanText = [regex]::Replace($Text, '\x1b\[[0-9;]*[A-Za-z]', '')
    $patterns = @(
        '(?i)wandb run name\s*[:=]\s*(.+)',
        '(?i)wandb[: ]+run name\s*[:=]\s*(.+)'
    )

    foreach ($line in ($cleanText -split "`r?`n")) {
        $stripped = $line.Trim()
        foreach ($pattern in $patterns) {
            if ($stripped -match $pattern) {
                return $Matches[1].Trim()
            }
        }
    }

    return $null
}

function Get-FailureReason {
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

    $matches = [System.Collections.Generic.List[string]]::new()
    foreach ($rawLine in ($cleanText -split "`r?`n")) {
        $line = $rawLine.Trim()
        if (-not $line) {
            continue
        }

        foreach ($pattern in $patterns) {
            if ($line -match $pattern) {
                if (-not $matches.Contains($line)) {
                    $matches.Add($line)
                }
                break
            }
        }
    }

    if ($matches.Count -gt 0) {
        return ($matches -join ' | ')
    }

    return $null
}

function Write-StructuredStatus {
    param(
        [string]$Status,
        [string]$Server,
        [int]$GpuId,
        [double]$GpuUtil,
        [string]$TmuxName,
        [string]$CommandName,
        [string]$WandbRunName,
        [string]$ErrorReason
    )

    Write-Host "status: $Status"
    Write-Host "server: $Server"
    Write-Host "gpu_id: $GpuId"
    Write-Host "gpu_util: $([math]::Round($GpuUtil, 0))"
    Write-Host "tmux_name: $TmuxName"
    Write-Host "command_name: $CommandName"
    Write-Host "wandb_run_name: $WandbRunName"
    if ($null -ne $ErrorReason) {
        Write-Host "error_reason: $ErrorReason"
    }
}

if ([string]::IsNullOrWhiteSpace(($TRAIN_COMMAND -replace '\s', ''))) {
    [Console]::Error.WriteLine("Set TRAIN_COMMAND at the top of this script before running it.")
    exit 1
}

$selection = $null
if (-not [string]::IsNullOrWhiteSpace(($SSH_NAME -replace '\s', '')) -and -not [string]::IsNullOrWhiteSpace(($GPU_ID -replace '\s', ''))) {
    $selection = Find-PreferredGpuOrError -SshNameValue $SSH_NAME -GpuIdText $GPU_ID
    if ($null -eq $selection) {
        exit 1
    }
} else {
    $selection = Find-FirstFreeGpu
    if ($null -eq $selection) {
        [Console]::Error.WriteLine("No reachable free GPU found.")
        exit 1
    }
}

$preparedCommand = Prepare-TrainCommand -GpuId $selection.GpuId
$commandName = Get-CommandName -Command $preparedCommand
$sessionName = Get-AvailableTmuxSessionName -HostAlias $selection.HostAlias
$startResult = Start-RemoteTraining -HostAlias $selection.HostAlias -SessionName $sessionName -PreparedCommand $preparedCommand -DataDirProcessed $DATA_DIR_PROCESSED

if ($startResult.ExitCode -ne 0) {
    [Console]::Error.WriteLine("Failed to start tmux session '$sessionName' on $($selection.HostAlias).")
    exit 1
}

$startTime = Get-Date
$wandbRunName = $null
$failureReason = $null
$paneOutput = ''

while ($true) {
    if (((Get-Date) - $startTime).TotalSeconds -ge $PollTimeoutSeconds) {
        Write-StructuredStatus `
            -Status 'timeout' `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName '-' `
            -ErrorReason 'Timed out waiting for wandb run name'
        exit 1
    }

    Start-Sleep -Seconds $PollIntervalSeconds

    if (((Get-Date) - $startTime).TotalSeconds -ge $PollTimeoutSeconds) {
        Write-StructuredStatus `
            -Status 'timeout' `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName '-' `
            -ErrorReason 'Timed out waiting for wandb run name'
        exit 1
    }

    $paneResult = Capture-TmuxOutput -HostAlias $selection.HostAlias -SessionName $sessionName
    if ($paneResult.ExitCode -ne 0) {
        [Console]::Error.WriteLine("Failed to capture tmux output from $($selection.HostAlias):$sessionName")
        if ($paneResult.Output) {
            [Console]::Error.WriteLine($paneResult.Output)
        }
        exit 1
    }

    $paneOutput = $paneResult.Output
    $wandbRunName = Get-WandbRunName -Text $paneOutput
    if ($wandbRunName) {
        break
    }

    $failureReason = Get-FailureReason -Text $paneOutput
    if ($failureReason) {
        Write-StructuredStatus `
            -Status 'failed' `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName '-' `
            -ErrorReason $failureReason
        exit 1
    }

    if (((Get-Date) - $startTime).TotalSeconds -ge $PollTimeoutSeconds) {
        Write-StructuredStatus `
            -Status 'timeout' `
            -Server $selection.HostAlias `
            -GpuId $selection.GpuId `
            -GpuUtil $selection.GpuUtil `
            -TmuxName $sessionName `
            -CommandName $commandName `
            -WandbRunName '-' `
            -ErrorReason 'Timed out waiting for wandb run name'
        exit 1
    }
}

Write-StructuredStatus `
    -Status 'started' `
    -Server $selection.HostAlias `
    -GpuId $selection.GpuId `
    -GpuUtil $selection.GpuUtil `
    -TmuxName $sessionName `
    -CommandName $commandName `
    -WandbRunName $wandbRunName `
    -ErrorReason $null
