# GPU Snatcher - Windows/PowerShell
# 自动监视 GPU 服务器，占住空闲 GPU 显存，等待用户释放
#
# 用法: .\gpu_snatcher.ps1 [-DryRun]
#       占满后按 Enter 释放所有显存并退出

param(
    # ============================================================
    # 超参数 —— 修改这里即可
    # ============================================================

    # 要监视的服务器列表（数字后缀如 230，或全名如 zju_4090_230）
    # 留空表示监视 SSH config 中所有 zju_ 服务器
    [string[]]$TargetServers = @(),

    # 每组需要的 GPU 数量（必须在同一台服务器上）
    [int]$GpusPerGroup = 2,

    # 总共需要占几组（占满后不再抢新 GPU，但继续监视等待用户释放）
    [int]$NumGroups = 3,

    # 目标显存占用比例（占完后该卡总占用率 ≈ 此值）
    [double]$GpuMemoryTargetRatio = 0.9,

    # 安全余量（MiB），防止 CUDA context / 碎片化导致 OOM
    [int]$GpuMemoryHeadroomMib = 256,

    # 轮询间隔（秒）
    [int]$PollIntervalSeconds = 30,

    # SSH 连接超时（秒）
    [int]$ConnectTimeoutSeconds = 5,

    # SSH 命令超时（秒）
    [int]$SshCommandTimeoutSeconds = 15,

    # 内存使用阈值：低于此比例的 GPU 视为空闲（默认 0.1 = 10%）
    [double]$MemoryUsageThreshold = 0.1,

    # tmux 会话名前缀
    [string]$TmuxSessionPrefix = "gpu_snatch",

    # 远程 conda 环境名（需装有 PyTorch + CUDA）
    [string]$RemoteCondaEnv = "rr",

    # 远程工作目录（存放占显存脚本）
    [string]$RemoteWorkDir = "/tmp",

    # SSH config 路径
    [string]$SshConfigPath = "$HOME/.ssh/config",

    # Dry run 模式：只打印不执行（用于调试）
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# SSH 工具函数
# ============================================================

function ConvertTo-SshArgumentString {
    param([string[]]$Arguments)

    return [string]::Join(' ', ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }))
}

function Invoke-SshCommand {
    param(
        [string]$HostAlias,
        [string]$RemoteCommand,
        [int]$CommandTimeoutSeconds = $SshCommandTimeoutSeconds
    )

    if ($DryRun) {
        Write-Host "[DRY_RUN] ssh $HostAlias $RemoteCommand" -ForegroundColor DarkGray
        return [pscustomobject]@{ ExitCode = 0; Output = "" }
    }

    $sshArgs = @(
        '-o', 'BatchMode=yes',
        '-o', "ConnectTimeout=$ConnectTimeoutSeconds",
        '-o', 'StrictHostKeyChecking=accept-new',
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

# ============================================================
# 服务器发现
# ============================================================

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
                if ($pattern -match '^zju_' -and $pattern -notmatch '[*?]') {
                    if ($seen.Add($pattern)) {
                        $result.Add($pattern)
                    }
                }
            }
        }
    }

    return @($result | Sort-Object -Unique)
}

function Resolve-HostAlias {
    param([string]$SshNameValue)

    if ($null -eq $SshNameValue) {
        return ''
    }

    $trimmed = $SshNameValue.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        return ''
    }

    if ($trimmed -match '^zju_') {
        return $trimmed
    }

    if ($trimmed -match '^\d+$') {
        $found = ''
        if (Test-Path $SshConfigPath) {
            $lines = Get-Content -Path $SshConfigPath -ErrorAction SilentlyContinue
            foreach ($line in $lines) {
                if ($line -match '^\s*Host\s+(.+)$') {
                    foreach ($pattern in ($Matches[1] -split '\s+')) {
                        if ($pattern -match '^zju_' -and $pattern -match "${trimmed}$" -and $pattern -notmatch '[*?]') {
                            $found = $pattern
                            break
                        }
                    }
                }
                if ($found) { break }
            }
        }
        if ($found) {
            return $found
        }
        return "zju_4090_$trimmed"
    }

    return $trimmed
}

# ============================================================
# GPU 状态查询
# ============================================================

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

# ============================================================
# GPU 占显存脚本（Python，base64 编码后传到远程执行）
# ============================================================

function Build-GpuOccupyScriptBase64 {
    param(
        [string]$GpuIdsCsv,
        [double]$TargetRatio,
        [int]$HeadroomMib
    )

    # 这里内联生成 Python 脚本并 base64 编码
    # 为了跨平台兼容，直接用 PowerShell 生成脚本内容

    $scriptContent = @"
import torch
import sys
import time

gpu_ids_str = "$GpuIdsCsv"
target_ratio = $TargetRatio
headroom_mib = $HeadroomMib
headroom_bytes = int(headroom_mib * 1024 * 1024)

kept_tensors = []

for gpu_id_str in gpu_ids_str.split(","):
    gpu_id = int(gpu_id_str.strip())
    torch.cuda.set_device(gpu_id)
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)

    used_bytes = total_bytes - free_bytes
    target_bytes = int(total_bytes * target_ratio)
    needed_bytes = target_bytes - used_bytes - headroom_bytes

    used_gb = used_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    if needed_bytes <= 0:
        current_pct = used_bytes / total_bytes * 100
        print(f"GPU {gpu_id}: already at {current_pct:.1f}% ({used_gb:.2f}/{total_gb:.2f} GiB), skipping")
        sys.stdout.flush()
        continue

    max_safe = free_bytes - headroom_bytes
    alloc_bytes = min(needed_bytes, max_safe)

    if alloc_bytes <= 0:
        print(f"GPU {gpu_id}: insufficient free memory "
              f"(free={free_bytes/(1024**3):.2f} GiB, needed>{needed_bytes/(1024**3):.2f} GiB)")
        sys.stdout.flush()
        continue

    allocated = False
    while not allocated and alloc_bytes > 0:
        num_elements = alloc_bytes // 4
        try:
            t = torch.zeros(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")
            kept_tensors.append(t)
            _, after_free = torch.cuda.mem_get_info(gpu_id)
            actual_used = total_bytes - after_free
            print(f"GPU {gpu_id}: +{alloc_bytes/(1024**3):.2f} GiB allocated, "
                  f"{used_gb:.2f} -> {actual_used/(1024**3):.2f} GiB "
                  f"({actual_used/total_bytes*100:.1f}% used)")
            sys.stdout.flush()
            allocated = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                alloc_bytes = int(alloc_bytes * 0.8)
                print(f"GPU {gpu_id}: OOM, retrying with {alloc_bytes/(1024**3):.2f} GiB...")
                sys.stdout.flush()
            else:
                print(f"GPU {gpu_id}: error - {e}")
                sys.stdout.flush()
                break

if kept_tensors:
    print(f"Occupation active on {len(kept_tensors)} GPU(s). Waiting for termination...")
else:
    print("No tensors allocated. Still waiting to hold the session...")
sys.stdout.flush()

while True:
    time.sleep(10)
"@

    return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($scriptContent))
}

# ============================================================
# tmux 远程会话管理
# ============================================================

function Launch-GpuOccupation {
    param(
        [string]$HostAlias,
        [string]$SessionName,
        [string]$GpuIdsCsv
    )

    $occupyB64 = Build-GpuOccupyScriptBase64 -GpuIdsCsv $GpuIdsCsv -TargetRatio $GpuMemoryTargetRatio -HeadroomMib $GpuMemoryHeadroomMib

    if ($DryRun) {
        Write-Host "[DRY_RUN] Would launch GPU occupation on $HostAlias :" -ForegroundColor DarkGray
        Write-Host "         session=$SessionName gpus=$GpuIdsCsv" -ForegroundColor DarkGray
        Write-Host "         target_ratio=$GpuMemoryTargetRatio headroom=${GpuMemoryHeadroomMib}MiB" -ForegroundColor DarkGray
        return $true
    }

    $remoteScript = @'
set -euo pipefail

session_name="$1"
work_dir="$2"
conda_env="$3"
occupy_b64="$4"
gpu_ids_csv="$5"
target_ratio="$6"
headroom_mib="$7"

if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux is not installed on $(hostname)" >&2
    exit 1
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "ERROR: tmux session $session_name already exists on $(hostname)" >&2
    exit 1
fi

mkdir -p "$work_dir"

occupy_script="$work_dir/gpu_occupy_${session_name}.py"
echo "$occupy_b64" | base64 -d > "$occupy_script"

tmux new-session -d -s "$session_name"
tmux set-option -t "$session_name" remain-on-exit on
tmux new-window -t "$session_name" -n occupy
tmux kill-window -t "${session_name}:0" 2>/dev/null || true

tmux send-keys -t "${session_name}:occupy" "cd $work_dir" Enter
tmux send-keys -t "${session_name}:occupy" "source ~/.bashrc" Enter
tmux send-keys -t "${session_name}:occupy" 'eval "$(conda shell.bash hook)"' Enter
tmux send-keys -t "${session_name}:occupy" "conda activate $conda_env" Enter
tmux send-keys -t "${session_name}:occupy" "echo 'GPU occupation starting on GPUs: $gpu_ids_csv'" Enter
tmux send-keys -t "${session_name}:occupy" "echo \"Target ratio: $target_ratio, Headroom: ${headroom_mib}MiB\"" Enter
tmux send-keys -t "${session_name}:occupy" "python3 $occupy_script" Enter

echo "Launched tmux session '$session_name' on $(hostname) for GPUs: $gpu_ids_csv"
'@

    # 将远程脚本通过 stdin 传给 ssh bash -s
    $tempScript = [System.IO.Path]::GetTempFileName()
    try {
        $remoteScript | Out-File -FilePath $tempScript -Encoding UTF8 -NoNewline

        $sshArgs = @(
            '-o', 'BatchMode=yes',
            '-o', "ConnectTimeout=$ConnectTimeoutSeconds",
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ServerAliveInterval=5',
            '-o', 'ServerAliveCountMax=1',
            $HostAlias,
            'bash', '-s', '--',
            $SessionName,
            $RemoteWorkDir,
            $RemoteCondaEnv,
            $occupyB64,
            $GpuIdsCsv,
            [string]$GpuMemoryTargetRatio,
            [string]$GpuMemoryHeadroomMib
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

        [void]$process.Start()

        $stdinWriter = $process.StandardInput
        $stdinWriter.Write((Get-Content -Path $tempScript -Raw))
        $stdinWriter.Close()

        if (-not $process.WaitForExit($SshCommandTimeoutSeconds * 1000)) {
            try { $process.Kill() } catch {}
            Write-Host "  ERROR: SSH timed out launching occupation on $HostAlias" -ForegroundColor Red
            return $false
        }

        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()

        if ($process.ExitCode -ne 0) {
            Write-Host "  ERROR launching occupation on $HostAlias (exit=$($process.ExitCode))" -ForegroundColor Red
            if ($stderr) { Write-Host "  $stderr" -ForegroundColor Red }
            return $false
        }

        if ($stdout) { Write-Host "  $stdout" }
        return $true
    } catch {
        Write-Host "  ERROR: $_" -ForegroundColor Red
        return $false
    } finally {
        if ($process) { $process.Dispose() }
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

function Release-GpuOccupation {
    param(
        [string]$HostAlias,
        [string]$SessionName
    )

    if ($DryRun) {
        Write-Host "[DRY_RUN] Would kill tmux session $SessionName on $HostAlias" -ForegroundColor DarkGray
        return
    }

    Write-Host "  Releasing $SessionName on $HostAlias..."
    $result = Invoke-SshCommand -HostAlias $HostAlias -RemoteCommand "if tmux has-session -t '$SessionName' 2>/dev/null; then tmux kill-session -t '$SessionName' && echo 'Released $SessionName'; else echo 'Session $SessionName not found (already released?)'; fi"
    Write-Host "    $($result.Output)"
}

# ============================================================
# 全局状态：已占组追踪
# ============================================================

$script:OccupiedServer  = @{}   # group_id -> host_alias
$script:OccupiedGpus    = @{}   # group_id -> gpu_ids_csv
$script:OccupiedSession = @{}   # group_id -> tmux_session_name
$script:OccupiedCount   = 0

function Release-AllGpus {
    if ($script:OccupiedCount -eq 0) {
        return
    }

    Write-Host ""
    Write-Host "=== Releasing all occupied GPUs ==="
    for ($i = 1; $i -le $script:OccupiedCount; $i++) {
        if ($script:OccupiedServer.ContainsKey($i) -and $script:OccupiedSession.ContainsKey($i)) {
            Release-GpuOccupation -HostAlias $script:OccupiedServer[$i] -SessionName $script:OccupiedSession[$i]
        }
    }
    Write-Host "All GPUs released."
}

# Ctrl+C 处理
$consoleHandler = {
    Write-Host ""
    Write-Host "Received interrupt signal, cleaning up..."
    Release-AllGpus
    exit 0
}
[Console]::TreatControlCAsInput = $false
try {
    [Console]::CancelKeyPress += $consoleHandler
} catch {
    # Fallback: some PowerShell hosts don't support CancelKeyPress
}

# ============================================================
# 获取目标服务器列表
# ============================================================

function Get-TargetHosts {
    $allHosts = Get-ZjuHostsFromSshConfig -Path $SshConfigPath

    if ($TargetServers.Count -eq 0) {
        return @($allHosts | Sort-Object { [int]([regex]::Match($_, '\d+$').Value) } -Descending)
    }

    $result = [System.Collections.Generic.List[string]]::new()
    foreach ($target in $TargetServers) {
        $normalized = Resolve-HostAlias -SshNameValue $target
        foreach ($host in $allHosts) {
            if ($host -eq $normalized) {
                $result.Add($host)
                break
            }
        }
    }
    return @($result)
}

# ============================================================
# 主循环
# ============================================================

function Main {
    Write-Host "=== GPU Snatcher ==="
    Write-Host "  Target servers:   $(if ($TargetServers.Count -gt 0) { $TargetServers -join ', ' } else { '(all zju_ servers)' })"
    Write-Host "  GPUs per group:   $GpusPerGroup"
    Write-Host "  Total groups:     $NumGroups"
    Write-Host "  Memory target:    $([int]($GpuMemoryTargetRatio * 100))%"
    Write-Host "  Memory headroom:  ${GpuMemoryHeadroomMib} MiB"
    Write-Host "  Poll interval:    ${PollIntervalSeconds}s"
    Write-Host "  Tmux prefix:      $TmuxSessionPrefix"
    if ($DryRun) {
        Write-Host "  *** DRY RUN MODE ***" -ForegroundColor Yellow
    }
    Write-Host ""

    $targetHosts = Get-TargetHosts
    if ($targetHosts.Count -eq 0) {
        Write-Host "ERROR: No target servers found. Check TargetServers and SSH config." -ForegroundColor Red
        exit 1
    }

    Write-Host "  Monitoring $($targetHosts.Count) server(s):"
    foreach ($host in $targetHosts) {
        Write-Host "    - $host"
    }
    Write-Host ""

    $allOccupied = $false

    while ($true) {
        $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Host ""
        Write-Host "[$timestamp] === GPU Snatcher Status ==="
        Write-Host "  Occupied: $script:OccupiedCount / $NumGroups groups"

        # -------- 如果还没占满，扫描服务器找空闲 GPU --------
        if ($script:OccupiedCount -lt $NumGroups) {
            Write-Host "  Scanning for free GPUs..."

            $serversChecked = 0
            $serversOk = 0
            $serversDown = 0

            foreach ($host in $targetHosts) {
                if ([string]::IsNullOrWhiteSpace($host)) { continue }
                $serversChecked++

                $inventory = Get-HostGpuInventory -HostAlias $host

                if ($inventory.State -eq 'DOWN') {
                    $serversDown++
                    Write-Host "    $host : DOWN"
                    continue
                }

                $serversOk++
                $freeGpus = @($inventory.Gpus | Where-Object { $_.Status -eq 'FREE' })
                $freeCount = $freeGpus.Count
                $totalCount = $inventory.Gpus.Count

                if ($freeCount -gt 0) {
                    $idList = ($freeGpus | ForEach-Object { "GPU$($_.GpuId)($($_.UsagePercent)%)" }) -join ', '
                    Write-Host "    $host : $freeCount/$totalCount free ($idList)"
                } else {
                    Write-Host "    $host : $freeCount/$totalCount free (no free GPUs)"
                }

                # 检查是否有足够的空闲 GPU
                if ($freeCount -ge $GpusPerGroup) {
                    $selectedGpus = $freeGpus | Select-Object -First $GpusPerGroup
                    $selectedIds = ($selectedGpus | ForEach-Object { $_.GpuId }) -join ','
                    $selectedDesc = ($selectedGpus | ForEach-Object { "GPU$($_.GpuId)($($_.UsagePercent)%)" }) -join ', '

                    $script:OccupiedCount++
                    $groupId = $script:OccupiedCount
                    $sessionName = "${TmuxSessionPrefix}_${groupId}"

                    $script:OccupiedServer[$groupId] = $host
                    $script:OccupiedGpus[$groupId] = $selectedIds
                    $script:OccupiedSession[$groupId] = $sessionName

                    Write-Host ""
                    Write-Host "  >>> Occupying group $groupId : $host GPU$selectedIds ($selectedDesc)" -ForegroundColor Green

                    if (Launch-GpuOccupation -HostAlias $host -SessionName $sessionName -GpuIdsCsv $selectedIds) {
                        Write-Host "  >>> Group $groupId launched successfully" -ForegroundColor Green
                    } else {
                        Write-Host "  >>> ERROR: Failed to launch group $groupId on $host" -ForegroundColor Red
                        $script:OccupiedCount--
                        $script:OccupiedServer.Remove($groupId)
                        $script:OccupiedGpus.Remove($groupId)
                        $script:OccupiedSession.Remove($groupId)
                    }

                    if ($script:OccupiedCount -ge $NumGroups) {
                        break
                    }
                }
            }

            Write-Host "  Servers: $serversChecked checked, $serversOk OK, $serversDown DOWN"
        }

        # -------- 打印已占组详情 --------
        if ($script:OccupiedCount -gt 0) {
            Write-Host ""
            Write-Host "  --- Occupied Groups ---"
            for ($i = 1; $i -le $script:OccupiedCount; $i++) {
                $srv = $script:OccupiedServer[$i]
                $gpus = $script:OccupiedGpus[$i]
                $sess = $script:OccupiedSession[$i]
                Write-Host "  [$i] $srv GPU$gpus (tmux: $sess)"
            }
        }

        # -------- 占满了，等待用户交互 --------
        if ($script:OccupiedCount -ge $NumGroups) {
            if (-not $allOccupied) {
                $allOccupied = $true
                Write-Host ""
                Write-Host "==============================================" -ForegroundColor Cyan
                Write-Host "  All $NumGroups group(s) occupied!" -ForegroundColor Cyan
                Write-Host "  Press Enter to release all GPUs and exit..." -ForegroundColor Cyan
                Write-Host "  (Ctrl+C also releases GPUs safely)" -ForegroundColor Cyan
                Write-Host "==============================================" -ForegroundColor Cyan
            } else {
                Write-Host ""
                Write-Host "  Waiting... Press Enter to release all GPUs and exit." -ForegroundColor Cyan
            }

            # 阻塞等待用户按 Enter
            [Console]::ReadLine() | Out-Null
            Release-AllGpus
            Write-Host "Done. Exiting."
            exit 0
        }

        # -------- 还没占满，等一轮再查 --------
        Write-Host ""
        Write-Host "  Sleeping ${PollIntervalSeconds}s before next check..."
        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

Main
