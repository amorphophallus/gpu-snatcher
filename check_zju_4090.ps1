param(
    [string]$SshConfigPath = "$HOME/.ssh/config",
    [double]$MemoryUsageThreshold = 0.1,
    [int]$ConnectTimeoutSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ZjuHostsFromSshConfig {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "SSH config not found: $Path"
    }

    $hosts = [System.Collections.Generic.List[string]]::new()
    $lines = Get-Content -LiteralPath $Path

    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) {
            continue
        }

        if ($trimmed -match '^(?i)Host\s+(.+)$') {
            $patterns = $Matches[1] -split '\s+'
            foreach ($pattern in $patterns) {
                if ($pattern -like 'zju_4090_*' -and $pattern -notmatch '[*?]') {
                    $hosts.Add($pattern)
                }
            }
        }
    }

    return @($hosts | Sort-Object -Unique)
}

function Invoke-SshCommand {
    param(
        [string]$HostAlias,
        [string]$RemoteCommand,
        [int]$TimeoutSeconds
    )

    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$TimeoutSeconds",
        $HostAlias,
        $RemoteCommand
    )

    $output = & ssh @sshArgs 2>&1
    $exitCode = $LASTEXITCODE

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output   = ($output | Out-String).Trim()
    }
}

function Get-HostGpuStatus {
    param(
        [string]$HostAlias,
        [double]$Threshold,
        [int]$TimeoutSeconds
    )

    $query = 'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
    $result = Invoke-SshCommand -HostAlias $HostAlias -RemoteCommand $query -TimeoutSeconds $TimeoutSeconds

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

try {
    $hosts = @(Get-ZjuHostsFromSshConfig -Path $SshConfigPath)
} catch {
    Write-Error $_
    exit 1
}

if (-not $hosts -or $hosts.Count -eq 0) {
    Write-Host "No explicit zju_4090_* hosts found in $SshConfigPath" -ForegroundColor Yellow
    exit 1
}

$results = @(
foreach ($hostAlias in $hosts) {
    Get-HostGpuStatus -HostAlias $hostAlias -Threshold $MemoryUsageThreshold -TimeoutSeconds $ConnectTimeoutSeconds
}
)

$summaryRows = foreach ($item in $results) {
    $freeList = if ($item.Reachable -and $item.Gpus.Count -gt 0) {
        (($item.Gpus | Where-Object Available | ForEach-Object { "GPU{0}({1}%)" -f $_.Index, $_.GpuUtil }) -join ', ')
    } else {
        ''
    }

    [pscustomobject]@{
        Host       = $item.Host
        SSH        = if ($item.Reachable) { "OK" } else { "DOWN" }
        GPUs       = if ($item.Reachable) { "$($item.AvailableGpuCount)/$($item.TotalGpuCount) free" } else { "-" }
        FreeGPUIds = if ($freeList) { $freeList } else { "-" }
        Note       = if ($item.Reachable) {
            if ($item.TotalGpuCount -ne 8) { "Expected 8 GPUs, got $($item.TotalGpuCount)" } else { "" }
        } else {
            $item.Error
        }
    }
}

Write-Host ""
Write-Host "=== ZJU 4090 GPU Summary ===" -ForegroundColor Cyan
$summaryRows | Format-Table -AutoSize

foreach ($item in $results) {
    Write-Host ""
    if (-not $item.Reachable) {
        Write-Host "[$($item.Host)] SSH unreachable" -ForegroundColor Red
        Write-Host "  $($item.Error)"
        continue
    }

    Write-Host "[$($item.Host)] $($item.AvailableGpuCount)/$($item.TotalGpuCount) GPUs free" -ForegroundColor Green
    $item.Gpus |
        Select-Object `
            @{Name='GPU'; Expression = { $_.Index }}, `
            @{Name='Status'; Expression = { $_.Status }}, `
            @{Name='Used(MiB)'; Expression = { $_.MemoryUsed }}, `
            @{Name='Total(MiB)'; Expression = { $_.MemoryTotal }}, `
            @{Name='MemUsage%'; Expression = { $_.UsagePercent }}, `
            @{Name='GpuUtil%'; Expression = { $_.GpuUtil }} |
        Format-Table -AutoSize
}

Write-Host ""
$usable = @($results | Where-Object Reachable | Where-Object { $_.AvailableGpuCount -gt 0 })
Write-Host "=== Recommended Targets ===" -ForegroundColor Cyan
if ($usable.Count -eq 0) {
    Write-Host "No currently usable GPUs found." -ForegroundColor Yellow
} else {
    foreach ($item in $usable) {
        $freeIds = ($item.Gpus | Where-Object Available | Sort-Object GpuUtil, Index | ForEach-Object { "GPU{0}({1}%)" -f $_.Index, $_.GpuUtil }) -join ', '
        Write-Host "$($item.Host): $freeIds"
    }
}
