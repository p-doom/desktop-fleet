from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_fleet.readiness import int_metadata, readiness_summary
from desktop_fleet.registry import upsert_registry
from desktop_fleet.spec import FleetRunLayout, make_server_specs


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_readiness_aggregates_registry_and_pool_status(tmp_path):
    registry_path = tmp_path / "registry.json"
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "desktop_pool" / "status",
    )[0]
    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata={
            "expected_env_servers": 1,
            "expected_ready_sessions": 2,
        },
        servers=[server],
    )
    status_dir = tmp_path / "desktop_pool" / "status"
    server_status_dir = Path(server.pool_status_dir)
    server_status_dir.mkdir(parents=True)
    (server_status_dir / "worker-a.json").write_text(
        json.dumps({"closed": False, "ready": 2, "starting": 0}),
        encoding="utf-8",
    )

    summary = readiness_summary(
        SimpleNamespace(
            registry=registry_path,
            status_dir=status_dir,
            pool_status_dir=None,
            run_root=registry_path.parent,
            min_ready_sessions=-1,
            expected_servers=0,
        )
    )

    assert summary["registry_ready"] is True
    assert summary["min_ready"] == 2
    assert summary["ready"] == 2
    assert summary["server_summaries"][0]["ready"] == 2


def test_readiness_derives_status_dir_from_registry_layout(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "osworld_rl",
    )
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
        pool_status_root=layout.pool_status_dir,
    )[0]
    upsert_registry(
        path=layout.registry_path,
        run_id="run",
        metadata={
            "layout": layout.as_metadata(),
            "expected_env_servers": 1,
            "expected_ready_sessions": 1,
        },
        servers=[server],
    )
    server_status_dir = Path(server.pool_status_dir)
    server_status_dir.mkdir(parents=True)
    (server_status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "ready": 1,
                "total_failed": 3,
                "retry_scheduled": True,
                "consecutive_start_failures": 2,
                "startup_cooldown_remaining_s": 12.5,
            }
        ),
        encoding="utf-8",
    )

    summary = readiness_summary(
        SimpleNamespace(
            registry=layout.registry_path,
            status_dir=None,
            pool_status_dir=None,
            run_root=layout.run_root,
            min_ready_sessions=-1,
            expected_servers=0,
        )
    )

    assert summary["registry_ready"] is True
    assert summary["status_dir"] == str(layout.pool_status_dir)
    assert summary["ready"] == 1
    assert summary["total_failed"] == 3
    assert summary["retry_scheduled_workers"] == 1
    assert summary["cooling_down_workers"] == 1
    assert summary["consecutive_start_failures"] == 2
    assert summary["startup_cooldown_remaining_s"] == 12.5
    assert summary["server_summaries"][0]["name"] == "osworld-0000"
    assert summary["server_summaries"][0]["total_failed"] == 3


def test_readiness_summary_ignores_stale_status_files(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "osworld_rl",
    )
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
        pool_status_root=layout.pool_status_dir,
    )[0]
    upsert_registry(
        path=layout.registry_path,
        run_id="run",
        metadata={
            "layout": layout.as_metadata(),
            "expected_env_servers": 1,
            "expected_ready_sessions": 1,
        },
        servers=[server],
    )
    server_status_dir = Path(server.pool_status_dir)
    server_status_dir.mkdir(parents=True)
    now = time.time()
    (server_status_dir / "worker-stale.json").write_text(
        json.dumps({"closed": False, "updated_at": now - 1000.0, "ready": 5}),
        encoding="utf-8",
    )
    (server_status_dir / "worker-fresh.json").write_text(
        json.dumps({"closed": False, "updated_at": now, "ready": 1}),
        encoding="utf-8",
    )

    summary = readiness_summary(
        SimpleNamespace(
            registry=layout.registry_path,
            status_dir=None,
            pool_status_dir=None,
            run_root=layout.run_root,
            min_ready_sessions=-1,
            expected_servers=0,
            status_stale_after_s=120.0,
        )
    )

    assert summary["ready"] == 1
    assert summary["active_status_files"] == 1
    assert summary["stale_status_files"] == 1
    assert summary["server_summaries"][0]["stale_status_files"] == 1


def test_int_metadata_rejects_a_malformed_registry_value():
    assert int_metadata({}, "expected_ready_sessions", default=7) == 7
    with pytest.raises(ValueError, match="expected_ready_sessions"):
        int_metadata({"expected_ready_sessions": "lots"}, "expected_ready_sessions", default=7)
