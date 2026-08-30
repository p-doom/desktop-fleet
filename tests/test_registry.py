from __future__ import annotations

import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

from desktop_fleet.registry import read_registry, upsert_registry
from desktop_fleet.spec import FleetRunLayout, make_server_specs
from desktop_fleet.supervise import read_registry_for_status, registry_path_for_args


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("ENV_FLEET_RUNTIME_ENV_FILE", "")


def test_registry_upsert_merges_by_server_name(tmp_path, environment_metadata):
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
        name_prefix="fixture",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "status",
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
        name_prefix="fixture",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "status",
    )

    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata=environment_metadata,
        servers=first,
    )
    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata=environment_metadata,
        servers=second,
    )

    registry = read_registry(registry_path)
    assert [server.name for server in registry.servers] == [
        "fixture-0000",
        "fixture-0001",
    ]
    assert registry.metadata["environment"] == environment_metadata["environment"]
    json.dumps(registry.as_dict())


def test_registry_rejects_a_conflicting_run_writer(tmp_path, environment_metadata):
    path = tmp_path / "registry.json"
    servers = node_specs(tmp_path, 0, replica_count=1)
    upsert_registry(
        path=path,
        run_id="run-a",
        metadata=environment_metadata,
        servers=servers,
    )

    with pytest.raises(ValueError, match="run_id is immutable"):
        upsert_registry(
            path=path,
            run_id="run-b",
            metadata=environment_metadata,
            servers=servers,
        )


def test_registry_rejects_conflicting_shared_metadata(tmp_path, environment_metadata):
    path = tmp_path / "registry.json"
    servers = node_specs(tmp_path, 0, replica_count=1)
    upsert_registry(
        path=path,
        run_id="run",
        metadata=environment_metadata,
        servers=servers,
    )
    conflicting = deepcopy(environment_metadata)
    conflicting["environment"]["source"]["name"] = "other"

    with pytest.raises(ValueError, match="shared metadata is immutable"):
        upsert_registry(
            path=path,
            run_id="run",
            metadata=conflicting,
            servers=servers,
        )


def node_service_metadata(node_rank: int) -> dict:
    return {
        "node_services": {
            "mock_service": {
                "nodes": {str(node_rank): {"path": f"/runtime/node-{node_rank}.json"}}
            }
        }
    }


def node_specs(tmp_path, node_rank: int, replica_count: int):
    return make_server_specs(
        host=f"node{node_rank:03d}",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=node_rank,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=replica_count,
        replica_offset=node_rank,
        name_prefix="fixture",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "status",
    )


def test_registry_upsert_keeps_every_node_leaf_of_a_shared_nested_key(
    tmp_path,
    environment_metadata,
):
    registry_path = tmp_path / "registry.json"

    for node_rank in (0, 1):
        upsert_registry(
            path=registry_path,
            run_id="run",
            metadata={**environment_metadata, **node_service_metadata(node_rank)},
            servers=node_specs(tmp_path, node_rank, replica_count=2),
        )

    registry = read_registry(registry_path)
    assert registry.metadata["node_services"]["mock_service"]["nodes"] == {
        "0": {"path": "/runtime/node-0.json"},
        "1": {"path": "/runtime/node-1.json"},
    }


def test_registry_rejects_conflicting_aggregated_writers(
    tmp_path,
    environment_metadata,
):
    path = tmp_path / "registry.json"
    servers = node_specs(tmp_path, 0, replica_count=1)
    upsert_registry(
        path=path,
        run_id="run",
        metadata={**environment_metadata, **node_service_metadata(0)},
        servers=servers,
    )
    conflicting = node_service_metadata(0)
    conflicting["node_services"]["mock_service"]["nodes"]["0"] = {
        "path": "/runtime/other.json"
    }

    with pytest.raises(ValueError, match="writer conflict"):
        upsert_registry(
            path=path,
            run_id="run",
            metadata={**environment_metadata, **conflicting},
            servers=servers,
        )


def test_registry_upsert_serializes_concurrent_node_writers(
    tmp_path,
    environment_metadata,
):
    """Every node upserts into one file at once; the flock is what keeps them all."""
    registry_path = tmp_path / "registry.json"
    node_count = 8
    rounds = 25
    start = threading.Barrier(node_count)
    failures: list[BaseException] = []

    def publish(node_rank: int) -> None:
        specs = node_specs(tmp_path, node_rank, replica_count=node_count)
        metadata = {**environment_metadata, **node_service_metadata(node_rank)}
        start.wait()
        try:
            for _ in range(rounds):
                upsert_registry(
                    path=registry_path,
                    run_id="run",
                    metadata=metadata,
                    servers=specs,
                )
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            failures.append(error)

    threads = [
        threading.Thread(target=publish, args=(node_rank,))
        for node_rank in range(node_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    registry = read_registry(registry_path)
    assert [server.name for server in registry.servers] == [
        f"fixture-{node_rank:04d}" for node_rank in range(node_count)
    ]
    assert registry.metadata["node_services"]["mock_service"]["nodes"] == {
        str(node_rank): {"path": f"/runtime/node-{node_rank}.json"}
        for node_rank in range(node_count)
    }


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
