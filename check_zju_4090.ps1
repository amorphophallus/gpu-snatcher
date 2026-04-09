param(
    [string]$SshConfigPath = "$HOME/.ssh/config",
    [double]$MemoryUsageThreshold = 0.1,
    [int]$ConnectTimeoutSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
    $hosts = [System.Collections.Generic.List[string]]::new()

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^(?i)\s*Host\s+(.+)$') {
            foreach ($pattern in ($Matches[1] -split '\s+')) {
                if ($pattern -match '^zju_4090_' -and $pattern -notmatch '[*?]') {
                    if ($seen.Add($pattern)) {
                        $hosts.Add($pattern)
                    }
                }
            }
        }
    }

    return @($hosts | Sort-Object)
}

function Invoke-SshQuery {
    param(
        [string]$HostAlias,
        [int]$TimeoutSeconds
    )

    $query = 'nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$TimeoutSeconds",
        '-o', 'StrictHostKeyChecking=accept-new',
        $HostAlias,
        $query
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = 'ssh'
    $psi.Arguments = ConvertTo-SshArgumentString -Arguments $sshArgs
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    try {
        [void]$process.Start()
        $waitMilliseconds = [math]::Max(1000, ($TimeoutSeconds + 2) * 1000)
        if (-not $process.WaitForExit($waitMilliseconds)) {
            try { $process.Kill() } catch {}
            return [pscustomobject]@{
                ExitCode = 124
                Output   = "SSH command timed out after $($TimeoutSeconds + 2) seconds."
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

function Get-HostCheckRows {
    param(
        [string]$HostAlias,
        [double]$Threshold,
        [int]$TimeoutSeconds
    )

    $result = Invoke-SshQuery -HostAlias $HostAlias -TimeoutSeconds $TimeoutSeconds

    if ($result.ExitCode -ne 0) {
        return @(
            [pscustomobject]@{
                RowType = 'HOST'
                Host    = $HostAlias
                Field1  = 'DOWN'
                Field2  = if ($result.Output) { $result.Output } else { "SSH failed with exit code $($result.ExitCode)" }
                Field3  = ''
                Field4  = ''
                Field5  = ''
                Field6  = ''
            }
        )
    }

    $rows = [System.Collections.Generic.List[object]]::new()
    $rows.Add([pscustomobject]@{
        RowType = 'HOST'
        Host    = $HostAlias
        Field1  = 'OK'
        Field2  = ''
        Field3  = ''
        Field4  = ''
        Field5  = ''
        Field6  = ''
    })

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
        $usagePercent = [math]::Round($usageRatio * 100, 1)
        $status = if ($usageRatio -lt $Threshold) { 'FREE' } else { 'BUSY' }

        $rows.Add([pscustomobject]@{
            RowType = 'GPU'
            Host    = $HostAlias
            Field1  = [string]$index
            Field2  = $status
            Field3  = [string][int][math]::Round($memoryUsed)
            Field4  = [string][int][math]::Round($memoryTotal)
            Field5  = [string]$usagePercent
            Field6  = [string][int][math]::Round($gpuUtil)
        })
    }

    return @($rows)
}

function Get-HostStateOrDefault {
    param([string]$HostAlias)

    if ($hostState.ContainsKey($HostAlias)) {
        return $hostState[$HostAlias]
    }

    return 'DOWN'
}

try {
    $hosts = @(Get-ZjuHostsFromSshConfig -Path $SshConfigPath)
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}

if (-not $hosts -or $hosts.Count -eq 0) {
    [Console]::Error.WriteLine("No explicit zju_4090_* hosts found in $SshConfigPath")
    exit 1
}

$hostState = @{}
$hostNote = @{}
$hostFreeIds = @{}
$hostFreeCount = @{}
$hostTotalCount = @{}
$gpuRows = @{}

foreach ($hostAlias in $hosts) {
    foreach ($row in (Get-HostCheckRows -HostAlias $hostAlias -Threshold $MemoryUsageThreshold -TimeoutSeconds $ConnectTimeoutSeconds)) {
        if ($row.RowType -eq 'HOST') {
            $hostState[$row.Host] = $row.Field1
            $hostNote[$row.Host] = $row.Field2
            $hostFreeCount[$row.Host] = 0
            $hostTotalCount[$row.Host] = 0
            $hostFreeIds[$row.Host] = ''
            $gpuRows[$row.Host] = ''
            continue
        }

        $gpuIndex = $row.Field1
        $gpuStatus = $row.Field2
        $usedMib = $row.Field3
        $totalMib = $row.Field4
        $usagePercent = $row.Field5
        $gpuUtil = $row.Field6

        $hostTotalCount[$row.Host] = [int]$hostTotalCount[$row.Host] + 1
        $gpuRows[$row.Host] += "{0,-4} {1,-6} {2,-10} {3,-10} {4,-10} {5,-8}`n" -f $gpuIndex, $gpuStatus, $usedMib, $totalMib, $usagePercent, $gpuUtil

        if ($gpuStatus -eq 'FREE') {
            $hostFreeCount[$row.Host] = [int]$hostFreeCount[$row.Host] + 1
            if ($hostFreeIds[$row.Host]) {
                $hostFreeIds[$row.Host] += ', '
            }
            $hostFreeIds[$row.Host] += "GPU${gpuIndex}(${gpuUtil}%)"
        }
    }
}

Write-Host ""
Write-Host "=== ZJU 4090 GPU Summary ==="
Write-Host ("{0,-16} {1,-6} {2,-12} {3,-16} {4}" -f 'Host', 'SSH', 'GPUs', 'FreeGPUIds', 'Note')
foreach ($hostAlias in $hosts) {
    $note = ''
    $gpuSummary = '-'
    $freeIds = '-'
    $state = Get-HostStateOrDefault -HostAlias $hostAlias

    if ($state -eq 'OK') {
        $gpuSummary = "$($hostFreeCount[$hostAlias])/$($hostTotalCount[$hostAlias]) free"
        if ($hostFreeIds[$hostAlias]) {
            $freeIds = $hostFreeIds[$hostAlias]
        }
        if ([int]$hostTotalCount[$hostAlias] -ne 8) {
            $note = "Expected 8 GPUs, got $($hostTotalCount[$hostAlias])"
        }
    } else {
        $note = $hostNote[$hostAlias]
    }

    Write-Host ("{0,-16} {1,-6} {2,-12} {3,-16} {4}" -f $hostAlias, $state, $gpuSummary, $freeIds, $note)
}

foreach ($hostAlias in $hosts) {
    Write-Host ""
    if ((Get-HostStateOrDefault -HostAlias $hostAlias) -ne 'OK') {
        Write-Host "[$hostAlias] SSH unreachable"
        Write-Host "  $($hostNote[$hostAlias])"
        continue
    }

    Write-Host "[$hostAlias] $($hostFreeCount[$hostAlias])/$($hostTotalCount[$hostAlias]) GPUs free"
    Write-Host ("{0,-4} {1,-6} {2,-10} {3,-10} {4,-10} {5,-8}" -f 'GPU', 'Status', 'Used(MiB)', 'Total(MiB)', 'MemUsage%', 'GpuUtil%')
    if ($gpuRows[$hostAlias]) {
        Write-Host -NoNewline $gpuRows[$hostAlias]
    }
}

Write-Host ""
Write-Host "=== Recommended Targets ==="
$foundUsable = $false
foreach ($hostAlias in $hosts) {
    if (((Get-HostStateOrDefault -HostAlias $hostAlias) -eq 'OK') -and ([int]$hostFreeCount[$hostAlias] -gt 0)) {
        Write-Host "${hostAlias}: GPU $($hostFreeIds[$hostAlias])"
        $foundUsable = $true
    }
}

if (-not $foundUsable) {
    Write-Host "No currently usable GPUs found."
}
