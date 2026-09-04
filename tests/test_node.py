from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import desktop_fleet.node as node_module
from desktop_fleet.node import run_node
from desktop_fleet.registry import read_registry, upsert_registry


class RecordingService:
    def __init__(
        self,
        *,
        started_path: Path,
        closed_path: Path,
        metadata: dict[str, Any],
        fail_when_path_exists: Path | None = None,
    ) -> None:
        self.started_path = started_path
        self.closed_path = closed_path
        self.metadata = metadata
        self.fail_when_path_exists = fail_when_path_exists
        self.start_count = 0
        self.health_checks = 0

    def start(self) -> None:
        self.start_count += 1
        self.started_path.write_text("started", encoding="utf-8")

    def registry_metadata(self) -> dict[str, Any]:
        return self.metadata

    def assert_alive(self) -> None:
        self.health_checks += 1
        if (
            self.fail_when_path_exists is not None
            and self.fail_when_path_exists.exists()
        ):
            raise RuntimeError("node service failed")

    def close(self) -> None:
        self.closed_path.write_text("closed", encoding="utf-8")


def python_command(source: str, *args: Path) -> list[str]:
    return [sys.executable, "-c", source, *(str(arg) for arg in args)]


def service_for_rank(tmp_path: Path, node_rank: int) -> RecordingService:
    return RecordingService(
        started_path=tmp_path / f"started-{node_rank}",
        closed_path=tmp_path / f"closed-{node_rank}",
        metadata={
            "mock_service": {
                "nodes": {str(node_rank): {"path": f"/runtime/node-{node_rank}.json"}}
            }
        },
    )


def seed_registry(path: Path, environment_metadata: dict[str, object]) -> None:
    upsert_registry(
        path=path,
        run_id="run",
        metadata=environment_metadata,
        servers=(),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": ""}, "run_id"),
        ({"prepare_command": []}, "prepare_command"),
        ({"node_command": []}, "node_command"),
    ],
)
def test_invalid_node_contract_fails_before_starting_the_service(
    tmp_path, overrides, message
):
    service = service_for_rank(tmp_path, 0)
    arguments = {
        "registry_path": tmp_path / "registry.json",
        "run_id": "run",
        "prepare_command": python_command("pass"),
        "node_command": python_command("pass"),
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        run_node(service, **arguments)

    assert service.start_count == 0
    assert not service.closed_path.exists()


def test_node_service_starts_before_preparation_and_is_checked_until_node_exit(
    tmp_path, monkeypatch, environment_metadata
):
    monkeypatch.setattr(node_module, "_HEALTH_INTERVAL_S", 0.01)
    started = tmp_path / "started"
    prepared = tmp_path / "prepared"
    node_started = tmp_path / "node-started"
    closed = tmp_path / "closed"
    service = RecordingService(
        started_path=started,
        closed_path=closed,
        metadata={"service": {"nodes": {"0": {"path": "/service.json"}}}},
    )
    registry_path = tmp_path / "registry.json"
    seed_registry(registry_path, environment_metadata)

    returncode = run_node(
        service,
        registry_path=registry_path,
        run_id="run",
        prepare_command=python_command(
            "from pathlib import Path; "
            "started, prepared = map(Path, __import__('sys').argv[1:]); "
            "assert started.is_file(); prepared.write_text('prepared')",
            started,
            prepared,
        ),
        node_command=python_command(
            "from pathlib import Path; import sys, time; "
            "prepared, started = map(Path, sys.argv[1:]); "
            "assert prepared.is_file(); started.write_text('started'); time.sleep(0.08)",
            prepared,
            node_started,
        ),
    )

    assert returncode == 0
    assert service.start_count == 1
    assert service.health_checks >= 2
    assert node_started.is_file()
    assert closed.is_file()


def test_node_service_failure_terminates_the_node_process_and_closes(
    tmp_path,
    environment_metadata,
):
    node_pid = tmp_path / "node.pid"
    closed = tmp_path / "closed"
    service = RecordingService(
        started_path=tmp_path / "started",
        closed_path=closed,
        metadata={"service": {"nodes": {"0": {"path": "/service.json"}}}},
        fail_when_path_exists=node_pid,
    )
    registry_path = tmp_path / "registry.json"
    seed_registry(registry_path, environment_metadata)

    with pytest.raises(RuntimeError, match="node service failed"):
        run_node(
            service,
            registry_path=registry_path,
            run_id="run",
            prepare_command=python_command("pass"),
            node_command=python_command(
                "from pathlib import Path; import os, sys, time; "
                "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
                node_pid,
            ),
        )

    pid = int(node_pid.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert closed.is_file()


def test_preparation_failure_closes_the_service_without_publishing_metadata(tmp_path):
    registry_path = tmp_path / "registry.json"
    service = service_for_rank(tmp_path, 0)

    with pytest.raises(subprocess.CalledProcessError):
        run_node(
            service,
            registry_path=registry_path,
            run_id="run",
            prepare_command=python_command("raise SystemExit(7)"),
            node_command=python_command("raise AssertionError('must not start')"),
        )

    assert service.closed_path.is_file()
    assert not registry_path.exists()


def test_successful_preparation_must_publish_the_shared_registry(tmp_path):
    service = service_for_rank(tmp_path, 0)

    with pytest.raises(FileNotFoundError):
        run_node(
            service,
            registry_path=tmp_path / "registry.json",
            run_id="run",
            prepare_command=python_command("pass"),
            node_command=python_command("raise AssertionError('must not start')"),
        )

    assert service.closed_path.is_file()


def test_node_writers_preserve_each_others_descriptor_metadata(
    tmp_path,
    environment_metadata,
):
    registry_path = tmp_path / "registry.json"
    seed_registry(registry_path, environment_metadata)

    for node_rank in (0, 1):
        service = service_for_rank(tmp_path, node_rank)
        assert (
            run_node(
                service,
                registry_path=registry_path,
                run_id="run",
                prepare_command=python_command("pass"),
                node_command=python_command("pass"),
            )
            == 0
        )

    registry = read_registry(registry_path)
    assert registry.metadata["node_services"]["mock_service"]["nodes"] == {
        "0": {"path": "/runtime/node-0.json"},
        "1": {"path": "/runtime/node-1.json"},
    }
