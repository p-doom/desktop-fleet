"""Capacity accounting: how many leased machines are actually usable right now.

Worker pools heartbeat one JSON status file each. This module turns those files
plus the registry into counters (``ready``/``starting``/``leased``), discards
stale writers, and exposes the blocking ``main()`` gate a consumer waits on
before it starts asking the fleet for rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from env_fleet.registry import EnvFleetRegistry, read_registry_if_ready
from env_fleet.spec import FleetRunLayout, load_runtime_env_file


class ServerReadinessSummary(TypedDict):
    name: str
    address: str
    status_dir: str | None
    status_files: int
    active_status_files: int
    stale_status_files: int
    ready: int
    starting: int
    leased: int
    total_failed: int
    stale_leases_retired: int
    last_errors: list[str]


class ReadinessSummary(TypedDict):
    registry: str
    registry_ready: bool
    registry_error: str | None
    expected_servers: int
    registered_servers: int
    status_dir: str
    status_files: int
    active_status_files: int
    stale_status_files: int
    min_ready: int
    ready: int
    starting: int
    leased: int
    total_started: int
    total_failed: int
    stale_leases_retired: int
    retry_scheduled_workers: int
    cooling_down_workers: int
    consecutive_start_failures: int
    startup_cooldown_remaining_s: float
    last_errors: list[str]
    server_summaries: list[ServerReadinessSummary]
    unhealthy_servers: int


def readiness_summary(args: Any) -> ReadinessSummary:
    """Aggregate registry and worker status files into readiness counters."""
    registry, registry_error = read_registry_if_ready(args.registry)
    metadata = registry.metadata if registry is not None else {}
    expected_servers = args.expected_servers or int_metadata(
        metadata,
        "expected_env_servers",
        default=1,
    )
    server_count = len(registry.servers) if registry is not None else 0
    registry_ready = registry is not None and server_count >= expected_servers
    status_dir = resolve_status_dir(
        args.status_dir,
        args.run_root,
        metadata,
        pool_status_dir=args.pool_status_dir,
    )
    statuses = read_statuses(status_dir, recursive=True)
    status_stale_after_s = getattr(args, "status_stale_after_s", None)
    now = time.time()
    active_statuses = active_worker_statuses(
        statuses,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    stale_statuses = stale_worker_statuses(
        statuses,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    min_ready = resolve_min_ready(args, metadata)
    ready = sum_int_field(active_statuses, "ready")
    starting = sum_int_field(active_statuses, "starting")
    leased = sum_int_field(active_statuses, "leased")
    total_started = sum_int_field(active_statuses, "total_started")
    total_failed = sum_int_field(active_statuses, "total_failed")
    stale_leases_retired = sum_int_field(active_statuses, "stale_leases_retired")
    retry_scheduled_workers = sum_bool_field(active_statuses, "retry_scheduled")
    cooling_down_workers = count_positive_float_field(
        active_statuses,
        "startup_cooldown_remaining_s",
    )
    consecutive_start_failures = sum_int_field(
        active_statuses,
        "consecutive_start_failures",
    )
    startup_cooldown_remaining_s = max_float_field(
        active_statuses,
        "startup_cooldown_remaining_s",
    )
    last_errors = [
        str(status["last_error"])
        for status in active_statuses
        if status.get("last_error")
    ]
    server_summaries = summarize_registry_servers(
        registry,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    unhealthy_servers = sum(
        1
        for server in server_summaries
        if (
            server["active_status_files"] > 0
            and server["ready"] + server["leased"] <= 0
        )
    )

    return {
        "registry": str(args.registry),
        "registry_ready": registry_ready,
        "registry_error": registry_error,
        "expected_servers": expected_servers,
        "registered_servers": server_count,
        "status_dir": str(status_dir),
        "status_files": len(statuses),
        "active_status_files": len(active_statuses),
        "stale_status_files": len(stale_statuses),
        "min_ready": min_ready,
        "ready": ready,
        "starting": starting,
        "leased": leased,
        "total_started": total_started,
        "total_failed": total_failed,
        "stale_leases_retired": stale_leases_retired,
        "retry_scheduled_workers": retry_scheduled_workers,
        "cooling_down_workers": cooling_down_workers,
        "consecutive_start_failures": consecutive_start_failures,
        "startup_cooldown_remaining_s": startup_cooldown_remaining_s,
        "last_errors": last_errors[-5:],
        "server_summaries": server_summaries,
        "unhealthy_servers": unhealthy_servers,
    }


def read_statuses(status_dir: Path, *, recursive: bool = True) -> list[dict[str, Any]]:
    """Load all valid desktop-pool worker status JSON files from a directory."""
    if not status_dir.exists():
        return []
    statuses: list[dict[str, Any]] = []
    for path in status_paths(status_dir, recursive=recursive):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            statuses.append(payload)
    return statuses


def status_paths(status_dir: Path, *, recursive: bool) -> list[Path]:
    """Return current worker status JSON paths without descending into archives."""
    direct_paths = list(status_dir.glob("*.json"))
    if not recursive:
        return sorted(direct_paths)
    nested_paths = list(status_dir.glob("*/*.json"))
    return sorted({*direct_paths, *nested_paths})


def active_worker_statuses(
    statuses: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
    stale_after_s: float | None = None,
) -> list[Mapping[str, Any]]:
    """Filter out status files written by closed or stale worker pools."""
    current = time.time() if now is None else now
    return [
        status
        for status in statuses
        if not bool(status.get("closed"))
        and status_is_fresh(status, now=current, stale_after_s=stale_after_s)
    ]


def stale_worker_statuses(
    statuses: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
    stale_after_s: float | None = None,
) -> list[Mapping[str, Any]]:
    """Return non-closed worker status payloads older than the freshness window."""
    if stale_after_s is None:
        return []
    current = time.time() if now is None else now
    return [
        status
        for status in statuses
        if not bool(status.get("closed"))
        and not status_is_fresh(status, now=current, stale_after_s=stale_after_s)
    ]


def status_is_fresh(
    status: Mapping[str, Any],
    *,
    now: float,
    stale_after_s: float | None,
) -> bool:
    """Return whether a worker status payload is recent enough to trust."""
    if stale_after_s is None:
        return True
    try:
        updated_at = float(status["updated_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return now - updated_at <= stale_after_s


def summarize_registry_servers(
    registry: EnvFleetRegistry | None,
    *,
    now: float | None = None,
    stale_after_s: float | None = None,
) -> list[ServerReadinessSummary]:
    """Build per-replica status summaries from registry server metadata."""
    if registry is None:
        return []
    summaries: list[ServerReadinessSummary] = []
    for server in registry.servers:
        status_dir_value = server.pool_status_dir
        if status_dir_value:
            status_dir = Path(status_dir_value)
            statuses = read_statuses(status_dir, recursive=False)
        else:
            status_dir = None
            statuses = []
        current = time.time() if now is None else now
        active_statuses = active_worker_statuses(
            statuses,
            now=current,
            stale_after_s=stale_after_s,
        )
        stale_statuses = stale_worker_statuses(
            statuses,
            now=current,
            stale_after_s=stale_after_s,
        )
        last_errors = [
            str(status["last_error"])
            for status in active_statuses
            if status.get("last_error")
        ]
        summaries.append(
            {
                "name": server.name,
                "address": server.public_address,
                "status_dir": str(status_dir) if status_dir is not None else None,
                "status_files": len(statuses),
                "active_status_files": len(active_statuses),
                "stale_status_files": len(stale_statuses),
                "ready": sum_int_field(active_statuses, "ready"),
                "starting": sum_int_field(active_statuses, "starting"),
                "leased": sum_int_field(active_statuses, "leased"),
                "total_failed": sum_int_field(active_statuses, "total_failed"),
                "stale_leases_retired": sum_int_field(
                    active_statuses,
                    "stale_leases_retired",
                ),
                "last_errors": last_errors[-3:],
            }
        )
    return summaries


def resolve_min_ready(args: Any, metadata: Mapping[str, Any]) -> int:
    """Choose the explicit ready threshold or fall back to registry metadata."""
    if args.min_ready_sessions >= 0:
        return args.min_ready_sessions
    return int_metadata(metadata, "expected_ready_sessions", default=1)


def resolve_status_dir(
    status_dir: Path | None,
    run_root: Path,
    metadata: Mapping[str, Any],
    *,
    pool_status_dir: Path | None = None,
) -> Path:
    """Choose the explicit status dir or derive it from registry pool metadata."""
    if status_dir is not None:
        return status_dir
    layout = FleetRunLayout.from_metadata(metadata)
    if layout is not None:
        return layout.pool_status_dir
    if pool_status_dir is not None:
        return Path(pool_status_dir)
    return run_root.parent / "pool" / "status"


def int_metadata(metadata: Mapping[str, Any], key: str, *, default: int) -> int:
    if key not in metadata:
        return default
    value = metadata[key]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"registry metadata {key!r} is not an integer: {value!r}"
        ) from exc


def sum_int_field(statuses: Sequence[Mapping[str, Any]], field: str) -> int:
    total = 0
    for status in statuses:
        try:
            total += int(status.get(field, 0))
        except (TypeError, ValueError):
            continue
    return total


def sum_bool_field(statuses: Sequence[Mapping[str, Any]], field: str) -> int:
    """Count status payloads where a boolean-ish field is set."""
    return sum(1 for status in statuses if bool(status.get(field)))


def count_positive_float_field(
    statuses: Sequence[Mapping[str, Any]],
    field: str,
) -> int:
    total = 0
    for status in statuses:
        try:
            if float(status.get(field, 0.0)) > 0.0:
                total += 1
        except (TypeError, ValueError):
            continue
    return total


def max_float_field(statuses: Sequence[Mapping[str, Any]], field: str) -> float:
    max_value = 0.0
    for status in statuses:
        try:
            max_value = max(max_value, float(status.get(field, 0.0)))
        except (TypeError, ValueError):
            continue
    return max_value


def main(argv: Sequence[str] | None = None) -> int:
    """Poll fleet registry and pool status files until readiness is reached."""
    args = parse_args(argv)
    deadline = time.monotonic() + args.timeout_s

    while True:
        summary = readiness_summary(args)
        if summary["registry_ready"] and summary["ready"] >= summary["min_ready"]:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if time.monotonic() >= deadline:
            print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        if args.verbose:
            print(json.dumps(summary, sort_keys=True), flush=True)
        time.sleep(args.poll_s)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build CLI arguments with Slurm-friendly defaults from the environment."""
    load_runtime_env_file()
    env = os.environ
    layout = FleetRunLayout.from_env(env)
    parser = argparse.ArgumentParser(
        description="Wait for an env fleet registry and warm desktop pool readiness."
    )
    parser.add_argument("--run-root", type=Path, default=layout.run_root)
    parser.add_argument("--registry", type=Path, default=layout.registry_path)
    parser.add_argument(
        "--pool-status-dir",
        type=Path,
        default=layout.pool_status_dir,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_STATUS_DIR"])
            if "OSWORLD_DESKTOP_POOL_STATUS_DIR" in env
            else None
        ),
    )
    parser.add_argument(
        "--min-ready-sessions",
        type=int,
        default=int(env.get("OSWORLD_DESKTOP_POOL_MIN_READY_TOTAL", "-1")),
        help="Ready sessions required. -1 uses registry metadata.",
    )
    parser.add_argument(
        "--expected-servers",
        type=int,
        default=int(env.get("OSWORLD_EXPECTED_ENV_SERVERS", "0")),
    )
    parser.add_argument(
        "--status-stale-after-s",
        type=float,
        default=float(env.get("OSWORLD_STATUS_STALE_AFTER_S", "120")),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(env.get("OSWORLD_ENV_FLEET_READY_TIMEOUT", "3600")),
    )
    parser.add_argument(
        "--poll-s",
        type=float,
        default=float(env.get("OSWORLD_ENV_FLEET_READY_POLL", "5")),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
