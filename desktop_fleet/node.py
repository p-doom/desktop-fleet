"""Lifecycle for one fleet node and its node-local service."""

from __future__ import annotations

import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from desktop_fleet.registry import upsert_registry
from desktop_fleet.supervise import terminate_replica_process_group

_HEALTH_INTERVAL_S = 2.0
_TERMINATE_TIMEOUT_S = 30.0


class NodeService(Protocol):
    def start(self) -> None: ...

    def registry_metadata(self) -> Mapping[str, Any]: ...

    def assert_alive(self) -> None: ...

    def close(self) -> None: ...


class _StopRequestedError(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def run_node(
    service: NodeService,
    *,
    registry_path: Path,
    run_id: str,
    prepare_command: Sequence[str],
    node_command: Sequence[str],
) -> int:
    if not run_id:
        raise ValueError("run_id must not be empty")
    if not prepare_command:
        raise ValueError("prepare_command must not be empty")
    if not node_command:
        raise ValueError("node_command must not be empty")

    process: subprocess.Popen[bytes] | None = None
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        raise _StopRequestedError(signum)

    for signum in previous_handlers:
        signal.signal(signum, request_stop)

    try:
        service.start()
        metadata = service.registry_metadata()
        if not isinstance(metadata, Mapping) or not metadata:
            raise ValueError("node service registry metadata must not be empty")

        subprocess.run(prepare_command, check=True)
        upsert_registry(
            path=registry_path,
            run_id=run_id,
            metadata={"node_services": dict(metadata)},
            servers=(),
        )
        process = subprocess.Popen(node_command, start_new_session=True)
        while True:
            service.assert_alive()
            try:
                return process.wait(timeout=_HEALTH_INTERVAL_S)
            except subprocess.TimeoutExpired:
                pass
    except _StopRequestedError as stop:
        return 128 + stop.signum
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        terminate_replica_process_group(process, timeout_s=_TERMINATE_TIMEOUT_S)
        service.close()
