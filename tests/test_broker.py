"""Tests for the cross-node rollout broker.

Covers the supervisor's broker entrypoint, resolving cross-node backends out of
registry gateway metadata, capacity ranking, request routing, frontend
request-id preservation, health probing, backend quarantine on timeout,
pending-request queueing/expiry, and the capacity-reservation invariants that
stop the broker oversending a stale ready slot.
"""

from __future__ import annotations

import asyncio
import json
import time

import msgpack
import pytest

from env_fleet.broker import (
    NO_BACKEND_CAPACITY_TIMEOUT_ERROR,
    NO_HEALTHY_BACKENDS_ERROR,
    PENDING_QUEUE_FULL_ERROR,
    REQUEST_TIMEOUT_ERROR,
    GatewayBackend,
    ZMQRolloutGateway,
    available_ready_sessions,
    backend_capacity_rank,
    gateway_backend_addresses,
    gateway_backend_status_dirs,
    gateway_bind_address,
)
from env_fleet.spec import make_server_specs
from env_fleet.supervise import BROKER_MODULE, broker_command


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_supervisor_spawns_the_in_package_broker_entrypoint():
    assert broker_command("/usr/bin/python3") == [
        "/usr/bin/python3",
        "-m",
        BROKER_MODULE,
    ]
    assert BROKER_MODULE == "env_fleet.broker"


def test_broker_resolves_cross_node_backends_from_registry_gateway_metadata(tmp_path):
    metadata = {
        "gateway": {
            "bind_address": "tcp://0.0.0.0:5204",
            "public_address": "tcp://node001:5204",
            "backend_addresses": [
                "tcp://node001:5200",
                "tcp://node002:5202",
            ],
        }
    }
    servers = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=2,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "status",
    )

    addresses = gateway_backend_addresses(metadata, servers)

    assert gateway_bind_address(metadata) == "tcp://0.0.0.0:5204"
    assert addresses == ["tcp://node001:5200", "tcp://node002:5202"]
    assert gateway_backend_status_dirs(servers, addresses) == [
        str(tmp_path / "status" / "osworld-0000"),
        None,
    ]
    assert gateway_backend_addresses({}, servers) == ["tcp://node001:5200"]
    assert gateway_bind_address({}) is None


def test_backend_capacity_rank_prefers_a_free_leased_machine():
    free = GatewayBackend(
        index=0, address="tcp://node001:5200", status_dir="/status/0", ready_sessions=2
    )
    reserved = GatewayBackend(
        index=1,
        address="tcp://node002:5200",
        status_dir="/status/1",
        ready_sessions=2,
        reserved_ready_sessions=2,
    )

    assert available_ready_sessions(free) == 2
    assert available_ready_sessions(reserved) == 0
    assert backend_capacity_rank(free) == 0
    assert backend_capacity_rank(reserved) == 1


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []

    async def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append(frames)


def run(coro):
    return asyncio.run(coro)


# A gateway backend's status_dir is required (see broker.py: a backend with no
# status dir would report capacity_ready unconditionally). Tests that don't care
# about a given backend's capacity point it at a directory that never exists;
# read_statuses() treats a missing directory as "no status files".
UNUSED_STATUS_DIR = "/nonexistent/env-fleet-test-status-dir"


def gateway_with_fake_sockets(
    *,
    request_ids: list[bytes] | None = None,
    backend_status_dirs: list[str] | None = None,
    status_stale_after_s: float = 120.0,
) -> ZMQRolloutGateway:
    ids = iter(request_ids or [b"backend-request"])
    gateway = ZMQRolloutGateway(
        bind_address="tcp://127.0.0.1:6200",
        backend_addresses=[
            "tcp://127.0.0.1:5200",
            "tcp://127.0.0.1:5201",
        ],
        backend_status_dirs=backend_status_dirs
        or [UNUSED_STATUS_DIR, UNUSED_STATUS_DIR],
        status_stale_after_s=status_stale_after_s,
        request_id_factory=lambda: next(ids),
    )
    gateway.frontend = FakeSocket()
    for backend in gateway.backends:
        backend.socket = FakeSocket()
    return gateway


def write_pool_status(
    status_dir,
    *,
    ready: int,
    leased: int = 0,
    starting: int = 0,
    updated_at: float | None = None,
) -> None:
    status_dir.mkdir(exist_ok=True)
    (status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": time.time() if updated_at is None else updated_at,
                "ready": ready,
                "starting": starting,
                "leased": leased,
            }
        ),
        encoding="utf-8",
    )


def test_gateway_forwards_request_to_least_loaded_healthy_backend():
    gateway = gateway_with_fake_sockets(request_ids=[b"backend-1"])
    gateway.backends[0].healthy = True
    gateway.backends[0].in_flight = 3
    gateway.backends[1].healthy = True
    payload = b"\xffraw-rollout-payload"

    run(gateway.forward_request(b"client-a", b"front-1", payload))

    assert gateway.backends[0].socket.sent == []
    assert gateway.backends[1].socket.sent == [[b"backend-1", b"run_group", payload]]
    assert gateway.backends[1].in_flight == 1


def test_gateway_preserves_frontend_request_id_on_response():
    gateway = gateway_with_fake_sockets(request_ids=[b"backend-1"])
    gateway.backends[0].healthy = True
    response = msgpack.packb({"success": True, "output": None}, use_bin_type=True)

    run(gateway.forward_request(b"client-a", b"front-1", b"payload"))
    run(gateway.handle_backend_message(gateway.backends[0], [b"backend-1", response]))

    assert gateway.frontend.sent == [[b"client-a", b"front-1", response]]
    assert gateway.backends[0].in_flight == 0


def test_gateway_rejects_three_frame_frontend_message():
    gateway = gateway_with_fake_sockets()

    run(gateway.handle_frontend_message([b"client-a", b"front-1", b"payload"]))

    assert gateway.frontend.sent == []
    assert all(backend.socket.sent == [] for backend in gateway.backends)


def test_gateway_returns_failed_response_when_no_backend_is_healthy():
    gateway = gateway_with_fake_sockets()

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload"]
        )
    )

    frames = gateway.frontend.sent[0]
    response = msgpack.unpackb(frames[2], raw=False)
    assert frames[:2] == [b"client-a", b"front-1"]
    assert response == {
        "success": False,
        "error": NO_HEALTHY_BACKENDS_ERROR,
    }


def test_gateway_health_fails_when_no_backend_is_live():
    gateway = gateway_with_fake_sockets()

    run(gateway.handle_frontend_message([b"client-a", b"health-1", b"health", b""]))

    frames = gateway.frontend.sent[0]
    response = msgpack.unpackb(frames[2], raw=False)
    assert frames[:2] == [b"client-a", b"health-1"]
    assert response == {
        "success": False,
        "error": NO_HEALTHY_BACKENDS_ERROR,
    }


def test_gateway_health_uses_liveness_not_spare_capacity(tmp_path):
    status_dir = tmp_path / "leased"
    status_dir.mkdir()
    (status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": time.time(),
                "ready": 0,
                "starting": 0,
                "leased": 1,
            }
        ),
        encoding="utf-8",
    )
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(status_dir), UNUSED_STATUS_DIR]
    )
    backend = gateway.backends[0]
    backend.healthy = True

    gateway.refresh_backend_capacity(backend, now=time.monotonic())
    run(gateway.handle_frontend_message([b"client-a", b"health-1", b"health", b""]))

    frames = gateway.frontend.sent[0]
    response = msgpack.unpackb(frames[2], raw=False)
    assert backend.capacity_ready is False
    assert gateway.select_backend() is None
    assert frames[:2] == [b"client-a", b"health-1"]
    assert response == {
        "success": True,
        "error": None,
    }


def test_gateway_backend_health_probe_uses_msgpack_payload():
    gateway = gateway_with_fake_sockets()

    run(gateway.poll_backend_health())

    backend = gateway.backends[0]
    request_id, method, payload = backend.socket.sent[0]
    assert request_id == b"health"
    assert method == b"health"
    assert msgpack.unpackb(payload, raw=False) == {}
    assert backend.pending_health is True

    response = msgpack.packb(
        {"success": True, "error": None},
        use_bin_type=True,
    )
    run(gateway.handle_backend_message(backend, [request_id, response]))

    assert backend.pending_health is False
    assert backend.healthy is True
    assert backend.last_error is None


def test_gateway_queues_request_when_all_live_backends_are_busy(tmp_path):
    left_status_dir = tmp_path / "left"
    right_status_dir = tmp_path / "right"
    write_pool_status(left_status_dir, ready=0, leased=1)
    write_pool_status(right_status_dir, ready=0, leased=1)
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(left_status_dir), str(right_status_dir)]
    )
    for backend in gateway.backends:
        backend.healthy = True
        gateway.refresh_backend_capacity(backend, now=time.monotonic(), force=True)

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload"]
        )
    )

    assert gateway.frontend.sent == []
    assert gateway.backends[0].socket.sent == []
    assert gateway.backends[1].socket.sent == []
    assert list(gateway.pending_frontend_requests) == [(b"client-a", b"front-1")]


def test_gateway_routes_queued_request_to_backend_that_frees_capacity(tmp_path):
    left_status_dir = tmp_path / "left"
    right_status_dir = tmp_path / "right"
    write_pool_status(left_status_dir, ready=0, leased=1)
    write_pool_status(right_status_dir, ready=0, leased=1)
    gateway = gateway_with_fake_sockets(
        request_ids=[b"backend-right"],
        backend_status_dirs=[str(left_status_dir), str(right_status_dir)],
    )
    for backend in gateway.backends:
        backend.healthy = True
        gateway.refresh_backend_capacity(backend, now=time.monotonic(), force=True)

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload"]
        )
    )
    write_pool_status(right_status_dir, ready=1, leased=0)
    run(gateway.drain_pending_requests())

    assert gateway.pending_frontend_requests == {}
    assert gateway.backends[0].socket.sent == []
    assert gateway.backends[1].socket.sent == [
        [b"backend-right", b"run_group", b"payload"]
    ]
    assert gateway.backends[1].ready_sessions == 1
    assert gateway.backends[1].reserved_ready_sessions == 1
    assert gateway.backends[1].capacity_ready is False


def test_gateway_capacity_reservation_prevents_oversending_stale_ready_slot(tmp_path):
    ready_status_dir = tmp_path / "ready"
    busy_status_dir = tmp_path / "busy"
    write_pool_status(ready_status_dir, ready=1)
    write_pool_status(busy_status_dir, ready=0, leased=1)
    gateway = gateway_with_fake_sockets(
        request_ids=[b"backend-first"],
        backend_status_dirs=[str(ready_status_dir), str(busy_status_dir)],
    )
    for backend in gateway.backends:
        backend.healthy = True
        gateway.refresh_backend_capacity(backend, now=time.monotonic(), force=True)

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload-1"]
        )
    )
    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-2", b"run_group", b"payload-2"]
        )
    )

    assert gateway.backends[0].socket.sent == [
        [b"backend-first", b"run_group", b"payload-1"]
    ]
    assert gateway.backends[1].socket.sent == []
    assert list(gateway.pending_frontend_requests) == [(b"client-a", b"front-2")]


def test_gateway_repeated_drain_does_not_oversend_stale_ready_slot(tmp_path):
    ready_status_dir = tmp_path / "ready"
    busy_status_dir = tmp_path / "busy"
    write_pool_status(ready_status_dir, ready=1)
    write_pool_status(busy_status_dir, ready=0, leased=1)
    gateway = gateway_with_fake_sockets(
        request_ids=[b"backend-first"],
        backend_status_dirs=[str(ready_status_dir), str(busy_status_dir)],
    )
    for backend in gateway.backends:
        backend.healthy = True
        gateway.refresh_backend_capacity(backend, now=time.monotonic(), force=True)
    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload-1"]
        )
    )
    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-2", b"run_group", b"payload-2"]
        )
    )

    run(gateway.drain_pending_requests())

    assert gateway.backends[0].socket.sent == [
        [b"backend-first", b"run_group", b"payload-1"]
    ]
    assert list(gateway.pending_frontend_requests) == [(b"client-a", b"front-2")]
    assert gateway.backends[0].reserved_ready_sessions == 1
    assert gateway.backends[0].capacity_ready is False


def test_gateway_times_out_queued_request_waiting_for_capacity(tmp_path):
    status_dir = tmp_path / "busy"
    write_pool_status(status_dir, ready=0, leased=1)
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(status_dir), UNUSED_STATUS_DIR]
    )
    gateway.capacity_wait_timeout_s = 1.0
    backend = gateway.backends[0]
    backend.healthy = True
    gateway.refresh_backend_capacity(backend, now=time.monotonic(), force=True)

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload"]
        )
    )
    request = next(iter(gateway.pending_frontend_requests.values()))
    object.__setattr__(request, "created_at", time.monotonic() - 2.0)
    run(gateway.expire_pending_requests())

    frames = gateway.frontend.sent[-1]
    response = msgpack.unpackb(frames[2], raw=False)
    assert frames[:2] == [b"client-a", b"front-1"]
    assert response["success"] is False
    assert NO_BACKEND_CAPACITY_TIMEOUT_ERROR in response["error"]
    assert gateway.pending_frontend_requests == {}


def test_gateway_rejects_request_when_pending_queue_is_full(tmp_path):
    status_dir = tmp_path / "busy"
    write_pool_status(status_dir, ready=0, leased=1)
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(status_dir), UNUSED_STATUS_DIR]
    )
    gateway.max_pending_requests = 1
    backend = gateway.backends[0]
    backend.healthy = True
    gateway.refresh_backend_capacity(backend, now=time.monotonic(), force=True)

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", b"payload-1"]
        )
    )
    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-2", b"run_group", b"payload-2"]
        )
    )

    frames = gateway.frontend.sent[-1]
    response = msgpack.unpackb(frames[2], raw=False)
    assert list(gateway.pending_frontend_requests) == [(b"client-a", b"front-1")]
    assert frames[:2] == [b"client-a", b"front-2"]
    assert response == {"success": False, "error": PENDING_QUEUE_FULL_ERROR}


def test_gateway_does_not_inspect_or_mutate_rollout_payload_contents():
    gateway = gateway_with_fake_sockets(request_ids=[b"backend-1"])
    gateway.backends[0].healthy = True
    payload = b"this is deliberately not msgpack"

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_group", payload]
        )
    )

    assert gateway.backends[0].socket.sent == [[b"backend-1", b"run_group", payload]]


def test_gateway_preserves_verifiers_v1_method_frame():
    gateway = gateway_with_fake_sockets(request_ids=[b"backend-1"])
    gateway.backends[0].healthy = True

    run(
        gateway.handle_frontend_message(
            [b"client-a", b"front-1", b"run_rollout", b"payload"]
        )
    )

    assert gateway.backends[0].socket.sent == [
        [b"backend-1", b"run_rollout", b"payload"]
    ]


def test_gateway_times_out_stuck_backend_request():
    gateway = gateway_with_fake_sockets(request_ids=[b"backend-1"])
    gateway.request_timeout_s = 1.0
    gateway.backend_quarantine_s = 30.0
    gateway.backends[0].healthy = True

    run(gateway.forward_request(b"client-a", b"front-1", b"payload"))
    route = next(iter(gateway.routes_by_frontend.values()))
    object.__setattr__(route, "created_at", time.monotonic() - 2.0)

    run(gateway.expire_routes())

    frames = gateway.frontend.sent[-1]
    response = msgpack.unpackb(frames[2], raw=False)
    assert frames[:2] == [b"client-a", b"front-1"]
    assert response["success"] is False
    assert REQUEST_TIMEOUT_ERROR in response["error"]
    assert gateway.routes_by_frontend == {}
    assert gateway.routes_by_backend == {}
    assert gateway.backends[0].healthy is False
    assert gateway.backends[0].quarantined_until > time.monotonic()


def test_gateway_uses_pool_status_capacity_for_backend_selection(tmp_path):
    empty_status_dir = tmp_path / "empty"
    ready_status_dir = tmp_path / "ready"
    empty_status_dir.mkdir()
    ready_status_dir.mkdir()
    updated_at = time.time()
    (empty_status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": updated_at,
                "ready": 0,
                "starting": 0,
                "leased": 0,
            }
        ),
        encoding="utf-8",
    )
    (ready_status_dir / "worker-b.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": updated_at,
                "ready": 1,
                "starting": 0,
                "leased": 0,
            }
        ),
        encoding="utf-8",
    )
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(empty_status_dir), str(ready_status_dir)]
    )
    for backend in gateway.backends:
        backend.healthy = True

    now = time.monotonic() + 10.0
    for backend in gateway.backends:
        gateway.refresh_backend_capacity(backend, now=now)

    assert gateway.backends[0].capacity_ready is False
    assert gateway.backends[1].capacity_ready is True
    assert gateway.select_backend() is gateway.backends[1]


def test_gateway_ignores_stale_pool_status_capacity(tmp_path):
    status_dir = tmp_path / "stale"
    status_dir.mkdir()
    (status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": time.time() - 1000.0,
                "ready": 1,
                "starting": 0,
                "leased": 0,
            }
        ),
        encoding="utf-8",
    )
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(status_dir), UNUSED_STATUS_DIR]
    )
    backend = gateway.backends[0]
    backend.healthy = True
    backend.created_at = time.monotonic() - 1000.0

    gateway.refresh_backend_capacity(backend, now=time.monotonic())

    assert backend.capacity_ready is False
    assert backend.stale_status_files == 1


def test_gateway_waits_for_ready_status_before_routing(tmp_path):
    status_dir = tmp_path / "warming"
    status_dir.mkdir()
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(status_dir), UNUSED_STATUS_DIR]
    )
    backend = gateway.backends[0]
    backend.healthy = True
    now = time.monotonic()
    backend.created_at = now

    gateway.refresh_backend_capacity(backend, now=now)

    assert backend.capacity_ready is False
    assert gateway.select_backend() is None


def test_gateway_does_not_count_starting_as_routable_capacity(tmp_path):
    status_dir = tmp_path / "starting"
    status_dir.mkdir()
    (status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": time.time(),
                "ready": 0,
                "starting": 1,
                "leased": 0,
            }
        ),
        encoding="utf-8",
    )
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(status_dir), UNUSED_STATUS_DIR]
    )
    backend = gateway.backends[0]
    backend.healthy = True
    now = time.monotonic()
    backend.created_at = now

    gateway.refresh_backend_capacity(backend, now=now)

    assert backend.capacity_ready is False


def test_gateway_does_not_route_to_leased_only_backend(tmp_path):
    leased_status_dir = tmp_path / "leased"
    ready_status_dir = tmp_path / "ready"
    leased_status_dir.mkdir()
    ready_status_dir.mkdir()
    updated_at = time.time()
    (leased_status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": updated_at,
                "ready": 0,
                "starting": 0,
                "leased": 1,
            }
        ),
        encoding="utf-8",
    )
    (ready_status_dir / "worker-b.json").write_text(
        json.dumps(
            {
                "closed": False,
                "updated_at": updated_at,
                "ready": 1,
                "starting": 0,
                "leased": 0,
            }
        ),
        encoding="utf-8",
    )
    gateway = gateway_with_fake_sockets(
        backend_status_dirs=[str(leased_status_dir), str(ready_status_dir)]
    )
    for backend in gateway.backends:
        backend.healthy = True
        gateway.refresh_backend_capacity(backend, now=time.monotonic() + backend.index)
    gateway.backends[1].in_flight = 5

    assert gateway.backends[0].capacity_ready is False
    assert gateway.backends[1].capacity_ready is True
    assert gateway.select_backend() is gateway.backends[1]
