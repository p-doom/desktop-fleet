from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from desktop_fleet.registry import read_registry, upsert_registry
from desktop_fleet.spec import FleetRunLayout, make_server_specs
from desktop_fleet.supervise import read_registry_for_status, registry_path_for_args


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_registry_upsert_merges_by_server_name(tmp_path):
    registry_path = tmp_path / "registry.json"
    first = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=2,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
    )
    second = make_server_specs(
        host="node002",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=1,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=2,
        replica_offset=1,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
    )

    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata={"env_id": "rl"},
        servers=first,
    )
    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata={"task_base_path": "/tasks"},
        servers=second,
    )

    registry = read_registry(registry_path)
    assert [server.name for server in registry.servers] == [
        "osworld-0000",
        "osworld-0001",
    ]
    assert registry.metadata["env_id"] == "rl"
    assert registry.metadata["task_base_path"] == "/tasks"
    json.dumps(registry.as_dict())


def test_registry_helpers_keep_missing_distinct_from_corrupt(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="fleet-a",
        run_base=tmp_path / "runs",
    )
    args = SimpleNamespace(
        registry=None,
        run_id=layout.run_id,
        run_base=layout.run_base,
    )

    assert registry_path_for_args(args) == layout.registry_path
    assert (
        registry_path_for_args(
            SimpleNamespace(
                registry=str(tmp_path / "custom-registry.json"),
                run_id=layout.run_id,
                run_base=layout.run_base,
            )
        )
        == tmp_path / "custom-registry.json"
    )
    assert read_registry(layout.registry_path, if_missing="none") is None

    registry, error = read_registry_for_status(layout.registry_path)
    assert registry is None
    assert error == f"missing: {layout.registry_path}"

    layout.registry_path.parent.mkdir(parents=True)
    layout.registry_path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_registry(layout.registry_path, if_missing="none")

    registry, error = read_registry_for_status(layout.registry_path)
    assert registry is None
    assert error is not None
    assert str(layout.registry_path) in error
