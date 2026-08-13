#!/usr/bin/env python3
"""Hold GPU memory and yield when another process from the target user appears."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--target-ratio", required=True, type=float)
    parser.add_argument("--headroom-mib", required=True, type=int)
    parser.add_argument("--yield-user", required=True)
    parser.add_argument("--poll-seconds", required=True, type=float)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--chunk-mib", type=int, default=256)
    return parser.parse_args()


def query_compute_pids(gpu_uuid: str) -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids: set[int] = set()
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 2 or fields[0] != gpu_uuid:
            continue
        try:
            pids.add(int(fields[1]))
        except ValueError:
            continue
    return pids


def pid_owner_uid(pid: int) -> int | None:
    try:
        return Path(f"/proc/{pid}").stat().st_uid
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def find_requester_pids(
    gpu_uuid: str,
    *,
    yield_uid: int,
    ignored_pids: set[int],
) -> set[int]:
    return {
        pid
        for pid in query_compute_pids(gpu_uuid)
        if pid not in ignored_pids and pid_owner_uid(pid) == yield_uid
    }


def write_state(path: Path, **payload: Any) -> None:
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.target_ratio < 1.0:
        raise ValueError("--target-ratio must be between zero and one")
    if args.headroom_mib < 256:
        raise ValueError("--headroom-mib must be at least 256")
    if args.poll_seconds < 0.1:
        raise ValueError("--poll-seconds must be at least 0.1")
    if args.chunk_mib < 16:
        raise ValueError("--chunk-mib must be at least 16")

    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    yield_uid = pwd.getpwnam(args.yield_user).pw_uid
    own_pid = os.getpid()
    state_base = {
        "pid": own_pid,
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": args.gpu_uuid,
        "yield_user": args.yield_user,
        "target_ratio": args.target_ratio,
        "headroom_mib": args.headroom_mib,
    }
    write_state(state_path, state="starting", **state_base)

    stop_reason: list[str] = []

    def request_stop(signum: int, _frame: Any) -> None:
        stop_reason.append(signal.Signals(signum).name.lower())

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    held_tensors: list[Any] = []
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected Python environment")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "the worker requires exactly one CUDA-visible GPU; "
                f"got {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        torch.empty(1, dtype=torch.uint8, device="cuda:0")

        requesters = find_requester_pids(
            args.gpu_uuid,
            yield_uid=yield_uid,
            ignored_pids={own_pid},
        )
        if requesters:
            raise RuntimeError(
                f"refusing to reserve GPU: {args.yield_user} already has GPU PIDs "
                f"{sorted(requesters)}"
            )

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        desired_free_bytes = max(
            int(total_bytes * (1.0 - args.target_ratio)),
            args.headroom_mib * MIB,
        )
        remaining_bytes = max(0, free_bytes - desired_free_bytes)
        chunk_limit = args.chunk_mib * MIB

        while remaining_bytes >= 16 * MIB:
            attempt_bytes = min(remaining_bytes, chunk_limit)
            while attempt_bytes >= 16 * MIB:
                try:
                    held_tensors.append(
                        torch.empty(
                            attempt_bytes,
                            dtype=torch.uint8,
                            device="cuda:0",
                        )
                    )
                    remaining_bytes -= attempt_bytes
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    attempt_bytes //= 2
            else:
                break

        free_after_bytes, total_after_bytes = torch.cuda.mem_get_info(0)
        held_bytes = sum(tensor.numel() * tensor.element_size() for tensor in held_tensors)
        used_ratio = (total_after_bytes - free_after_bytes) / total_after_bytes
        print(
            f"holding {held_bytes / MIB:.0f} MiB on physical GPU "
            f"{args.physical_gpu} ({args.gpu_uuid}); total usage={used_ratio:.1%}",
            flush=True,
        )
        write_state(
            state_path,
            state="holding",
            held_mib=round(held_bytes / MIB),
            observed_used_ratio=used_ratio,
            **state_base,
        )

        consecutive_query_failures = 0
        while not stop_reason:
            try:
                requesters = find_requester_pids(
                    args.gpu_uuid,
                    yield_uid=yield_uid,
                    ignored_pids={own_pid},
                )
                consecutive_query_failures = 0
            except (OSError, subprocess.SubprocessError) as exc:
                consecutive_query_failures += 1
                print(
                    f"nvidia-smi process query failed "
                    f"({consecutive_query_failures}/3): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if consecutive_query_failures >= 3:
                    stop_reason.append("nvidia_smi_query_failed")
                time.sleep(args.poll_seconds)
                continue
            if requesters:
                stop_reason.append(
                    f"yield_to_{args.yield_user}_pids_"
                    + "_".join(str(pid) for pid in sorted(requesters))
                )
                print(
                    f"detected {args.yield_user} GPU process(es) "
                    f"{sorted(requesters)}; releasing reservation",
                    flush=True,
                )
                break
            time.sleep(args.poll_seconds)

        reason = stop_reason[0] if stop_reason else "completed"
        held_tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        write_state(state_path, state="released", reason=reason, **state_base)
        print(f"GPU reservation released: {reason}", flush=True)
        return 0
    except Exception as exc:
        held_tensors.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        write_state(state_path, state="error", error=str(exc), **state_base)
        print(f"GPU reservation failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
