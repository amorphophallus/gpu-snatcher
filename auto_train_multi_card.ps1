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

function ConvertTo-CommandString {
    param([string[]]$Parts)

    return [string]::Join(' ', ($Parts | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }))
}

# Multi-card training command.
$global:TRAIN_COMMAND_PARTS = @(
    "torchrun",
    "--standalone",
    "--nproc_per_node=2",
    "-m",
    "src.train.bc_ddp",
    "+experiment=rgbd/diff_unet",
    "task=[one_leg,round_table,lamp]",
    "data.demo_source=rollout",
    "data.data_subset=500",
    "data.demo_outcome=success",
    "data.suffix=rgbd-skill",
    "training.batch_size=256",
    "training.num_epochs=3000",
    "training.steps_per_epoch=-1",
    "training.save_per_epoch=500",
    "wandb.project=multi-task-rgbd-skill-low-500",
    "wandb.mode=online",
    "randomness=low",
    "dryrun=false",
    "training.num_epochs=4000"
)
$global:TRAIN_COMMAND = ConvertTo-CommandString -Parts $global:TRAIN_COMMAND_PARTS
$global:SSH_NAME = "230"
$global:NUM_GPUs = 2
$global:GPU_ID = "0,1"
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

function Split-CommandTokens {
    param([string]$Command)

    if ([string]::IsNullOrWhiteSpace($Command)) {
        return @()
    }

    $tokens = [regex]::Matches($Command.Trim(), '(?:"(?:\\.|[^"])*"|''(?:\\.|[^''])*''|\S+)') | ForEach-Object Value
    $parts = [System.Collections.Generic.List[string]]::new()

    foreach ($token in $tokens) {
        if ((($token.StartsWith('"')) -and $token.EndsWith('"')) -or (($token.StartsWith("'")) -and $token.EndsWith("'"))) {
            $parts.Add($token.Substring(1, $token.Length - 2))
        } else {
            $parts.Add($token)
        }
    }

    return @($parts)
}

function Test-IsLeadingEnvAssignment {
    param([string]$Token)

    return ($Token -match '^[A-Za-z_][A-Za-z0-9_]*=.*$') -and
        (-not $Token.StartsWith('/')) -and
        (-not $Token.StartsWith('./'))
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

function Get-HostGpuInventory {
    param([string]$HostAlias)

    $result = Invoke-SshCommand -HostAlias $HostAlias -RemoteCommand 'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
    if ($result.ExitCode -ne 0) {
        return [pscustomobject]@{
            HostAlias = $HostAlias
            State     = 'DOWN'
            Note      = $result.Output
            Gpus      = @()
        }
    }

    $gpus = [System.Collections.Generic.List[object]]::new()
    foreach ($line in ($result.Output -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $parts = $line -split '\s*,\s*'
        if ($parts.Count -lt 4) {
            continue
        }

        $memoryTotal = [double]$parts[1]
        $memoryUsed = [double]$parts[2]
        $gpuUtil = [double]$parts[3]
        $usageRatio = if ($memoryTotal -gt 0) { $memoryUsed / $memoryTotal } else { 1.0 }
        $status = if ($usageRatio -lt $MemoryUsageThreshold) { 'FREE' } else { 'BUSY' }

        $gpus.Add([pscustomobject]@{
            GpuId        = [int]$parts[0]
            Status       = $status
            MemoryUsed   = $memoryUsed
            MemoryTotal  = $memoryTotal
            UsagePercent = [math]::Round($usageRatio * 100, 1)
            GpuUtil      = $gpuUtil
        })
    }

    return [pscustomobject]@{
        HostAlias = $HostAlias
        State     = 'OK'
        Note      = ''
        Gpus      = @($gpus)
    }
}

function Normalize-GpuIdList {
    param([string]$GpuIdText)

    if ([string]::IsNullOrWhiteSpace($GpuIdText)) {
        return @()
    }

    $seen = [System.Collections.Generic.HashSet[int]]::new()
    $result = [System.Collections.Generic.List[int]]::new()

    foreach ($rawPart in ($GpuIdText -split ',')) {
        $part = $rawPart.Trim()
        if (-not $part) {
            throw "GPU_ID must be a comma-separated list of non-negative integers without empty items."
        }
        if ($part -notmatch '^\d+$') {
            throw "GPU_ID entries must be non-negative integers, got '$part'."
        }

        $value = [int]$part
        if (-not $seen.Add($value)) {
            throw "GPU_ID contains a duplicate entry: $value."
        }

        $result.Add($value)
    }

    return @($result)
}

function Get-HostPriority {
    param([string]$HostAlias)

    $suffix = ''
    if ($HostAlias -match '([0-9]+)$') {
        $suffix = $Matches[1]
    }

    if ($suffix -and ($FAST_SERVER -contains $suffix)) {
        return 0
    }
    if ($suffix -and ($SLOW_SERVER -contains $suffix)) {
        return 2
    }

    return 1
}

function Get-HostNumber {
    param([string]$HostAlias)

    if ($HostAlias -match '([0-9]+)$') {
        return [int]$Matches[1]
    }

    return -1
}

function Get-SortedHostAliases {
    return @(
        Get-ZjuHostsFromSshConfig -Path $SshConfigPath |
            Sort-Object `
                @{ Expression = { Get-HostPriority -HostAlias $_ } }, `
                @{ Expression = { Get-HostNumber -HostAlias $_ }; Descending = $true }, `
                @{ Expression = { $_ } }
    )
}

function Select-MultiGpuTargetOnHost {
    param(
        [string]$HostAlias,
        [int]$NumGpus,
        [int[]]$PreferredGpuIds = @()
    )

    $inventory = Get-HostGpuInventory -HostAlias $HostAlias
    if ($inventory.State -ne 'OK') {
        return [pscustomobject]@{
            Status    = 'DOWN'
            HostAlias = $HostAlias
            Note      = $inventory.Note
        }
    }

    $reportedGpuIds = @($inventory.Gpus | Sort-Object GpuId | ForEach-Object { $_.GpuId })
    $freeGpus = @($inventory.Gpus | Where-Object { $_.Status -eq 'FREE' })
    $selected = @()

    if ($PreferredGpuIds.Count -gt 0) {
        $freeMap = @{}
        foreach ($gpu in $freeGpus) {
            $freeMap[$gpu.GpuId] = $gpu
        }

        $preferredSelection = [System.Collections.Generic.List[object]]::new()
        foreach ($gpuId in $PreferredGpuIds) {
            if ($freeMap.ContainsKey($gpuId)) {
                $preferredSelection.Add($freeMap[$gpuId])
                if ($preferredSelection.Count -eq $NumGpus) {
                    break
                }
            }
        }

        if ($preferredSelection.Count -ge $NumGpus) {
            $selected = @($preferredSelection | Select-Object -First $NumGpus)
        }
    }

    if ($selected.Count -lt $NumGpus) {
        $selected = @($freeGpus | Sort-Object GpuUtil, MemoryUsed, GpuId | Select-Object -First $NumGpus)
    }

    if ($selected.Count -lt $NumGpus) {
        return [pscustomobject]@{
            Status        = 'INSUFFICIENT'
            HostAlias     = $HostAlias
            FreeCount     = $freeGpus.Count
            RequiredCount = $NumGpus
        }
    }

    return [pscustomobject]@{
        Status    = 'OK'
        HostAlias = $HostAlias
        GpuIds    = @($selected | ForEach-Object { $_.GpuId })
        FreeCount = $freeGpus.Count
    }
}

function Find-MultiGpuTargetOrError {
    param(
        [string]$SshNameValue,
        [int]$NumGpus,
        [int[]]$PreferredGpuIds = @()
    )

    if ($NumGpus -le 0) {
        [Console]::Error.WriteLine("NUM_GPUs must be a positive integer, got '$NumGpus'.")
        return $null
    }

    if (-not [string]::IsNullOrWhiteSpace($SshNameValue)) {
        $hostAlias = "zju_4090_$SshNameValue"
        $selection = Select-MultiGpuTargetOnHost -HostAlias $hostAlias -NumGpus $NumGpus -PreferredGpuIds $PreferredGpuIds

        switch ($selection.Status) {
            'OK' {
                return $selection
            }
            'DOWN' {
                [Console]::Error.WriteLine("Preferred host '$hostAlias' is unreachable: $($selection.Note)")
                return $null
            }
            'INSUFFICIENT' {
                [Console]::Error.WriteLine("Host '$hostAlias' has only $($selection.FreeCount) free GPUs; need $($selection.RequiredCount).")
                return $null
            }
            default {
                [Console]::Error.WriteLine("Failed to select GPUs on host '$hostAlias'.")
                return $null
            }
        }
    }

    foreach ($hostAlias in (Get-SortedHostAliases)) {
        $selection = Select-MultiGpuTargetOnHost -HostAlias $hostAlias -NumGpus $NumGpus -PreferredGpuIds @()
        if ($selection.Status -eq 'OK') {
            return $selection
        }
    }

    [Console]::Error.WriteLine("No reachable server has $NumGpus free GPUs.")
    return $null
}

function Prepare-TrainCommand {
    param(
        [int]$NumGpus,
        [string]$GpuIdsCsv
    )

    $parts = Split-CommandTokens -Command $TRAIN_COMMAND
    if ($parts.Count -eq 0) {
        throw "TRAIN_COMMAND is empty."
    }

    $envParts = [System.Collections.Generic.List[string]]::new()
    $index = 0
    while ($index -lt $parts.Count -and (Test-IsLeadingEnvAssignment -Token $parts[$index])) {
        $key = $parts[$index].Split('=', 2)[0]
        if ($key -ne 'CUDA_VISIBLE_DEVICES') {
            $envParts.Add($parts[$index])
        }
        $index += 1
    }

    if ($index -ge $parts.Count) {
        throw "TRAIN_COMMAND must start with torchrun for auto_train_multi_card."
    }

    $commandParts = @($parts[$index..($parts.Count - 1)])
    if ([System.IO.Path]::GetFileName($commandParts[0]) -ne 'torchrun') {
        throw "TRAIN_COMMAND must start with torchrun for auto_train_multi_card."
    }

    $filteredParts = [System.Collections.Generic.List[string]]::new()
    $cursor = 0
    while ($cursor -lt $commandParts.Count) {
        $token = $commandParts[$cursor]
        if ($token -match '^training\.gpu_id=\S+$') {
            $cursor += 1
            continue
        }
        if ($token -eq 'training.gpu_id') {
            if ($cursor + 1 -lt $commandParts.Count) {
                $cursor += 2
            } else {
                $cursor += 1
            }
            continue
        }
        if ($token -eq '--nproc_per_node') {
            if ($cursor + 1 -lt $commandParts.Count) {
                $cursor += 2
            } else {
                $cursor += 1
            }
            continue
        }
        if ($token -like '--nproc_per_node=*') {
            $cursor += 1
            continue
        }

        $filteredParts.Add($token)
        $cursor += 1
    }

    $finalParts = [System.Collections.Generic.List[string]]::new()
    $finalParts.Add("CUDA_VISIBLE_DEVICES=$GpuIdsCsv")
    foreach ($envPart in $envParts) {
        $finalParts.Add($envPart)
    }
    $finalParts.Add($filteredParts[0])
    $finalParts.Add("--nproc_per_node=$NumGpus")
    for ($i = 1; $i -lt $filteredParts.Count; $i++) {
        $finalParts.Add($filteredParts[$i])
    }

    return ConvertTo-CommandString -Parts @($finalParts)
}

function Get-CommandName {
    param([string]$Command)

    $parts = [System.Collections.Generic.List[string]]::new()
    foreach ($token in (Split-CommandTokens -Command $Command)) {
        $parts.Add($token)
    }

    while ($parts.Count -gt 0 -and (Test-IsLeadingEnvAssignment -Token $parts[0])) {
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
        [int]$NumGpus,
        [string]$GpuIds,
        [string]$TmuxName,
        [string]$CommandName,
        [string]$WandbRunName,
        [string]$ErrorReason
    )

    Write-Host "status: $Status"
    Write-Host "server: $Server"
    Write-Host "num_gpus: $NumGpus"
    Write-Host "gpu_ids: $GpuIds"
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

$requestedGpuCount = 0
if (-not [int]::TryParse([string]$NUM_GPUs, [ref]$requestedGpuCount) -or $requestedGpuCount -le 0) {
    [Console]::Error.WriteLine("NUM_GPUs must be a positive integer, got '$NUM_GPUs'.")
    exit 1
}

try {
    $preferredGpuIds = @(Normalize-GpuIdList -GpuIdText $GPU_ID)
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}

$selection = Find-MultiGpuTargetOrError -SshNameValue $SSH_NAME -NumGpus $requestedGpuCount -PreferredGpuIds $preferredGpuIds
if ($null -eq $selection) {
    exit 1
}

$gpuIdsCsv = $selection.GpuIds -join ','
$preparedCommand = Prepare-TrainCommand -NumGpus $requestedGpuCount -GpuIdsCsv $gpuIdsCsv
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
            -NumGpus $requestedGpuCount `
            -GpuIds $gpuIdsCsv `
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
            -NumGpus $requestedGpuCount `
            -GpuIds $gpuIdsCsv `
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
            -NumGpus $requestedGpuCount `
            -GpuIds $gpuIdsCsv `
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
            -NumGpus $requestedGpuCount `
            -GpuIds $gpuIdsCsv `
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
    -NumGpus $requestedGpuCount `
    -GpuIds $gpuIdsCsv `
    -TmuxName $sessionName `
    -CommandName $commandName `
    -WandbRunName $wandbRunName `
    -ErrorReason $null
