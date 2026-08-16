"""Cross-node ZMQ broker that routes on leased-machine capacity.

``tcp://`` end to end: one ROUTER frontend bound on a routable address, one
DEALER per env-server replica on other nodes of the Slurm allocation, so it
load-balances across a multi-node fleet. ``ipc://``, which the ``verifiers``
broker uses per worker, cannot express that.

The routing key is not round-robin or in-flight count alone. It reads
desktop-pool capacity off the worker status files on disk --
:func:`available_ready_sessions` and :func:`backend_capacity_rank` -- so a
replica whose VM pool has no free machine is skipped, and requests queue
instead of failing.

Everything routed through here is an opaque payload. The broker never inspects,
parses, or rewrites a rollout.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import msgpack
import zmq
import zmq.asyncio

from desktop_fleet.readiness import (
    active_worker_statuses,
    read_statuses,
    stale_worker_statuses,
    sum_int_field,
)
from desktop_fleet.registry import read_registry
from desktop_fleet.spec import EnvServerSpec, FleetRunLayout, load_runtime_env_file

HEALTH_REQUEST_ID = b"health"
HEALTH_METHOD = b"health"
HEALTH_REQUEST_PAYLOAD = msgpack.packb({}, use_bin_type=True)
RUN_GROUP_METHOD = b"run_group"
NO_HEALTHY_BACKENDS_ERROR = "No healthy env-server replicas are available"
NO_BACKEND_CAPACITY_TIMEOUT_ERROR = (
    "No env-server desktop capacity became available before gateway timeout"
)
PENDING_QUEUE_FULL_ERROR = "Env-server desktop capacity queue is full"
REQUEST_TIMEOUT_ERROR = "Env-server replica did not answer before gateway timeout"
DispatchStatus = Literal["sent", "no_capacity", "no_live_backend"]

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


@dataclass
class GatewayBackend:
    index: int
    address: str
    status_dir: str
    socket: Any | None = None
    healthy: bool = False
    capacity_ready: bool = True
    ready_sessions: int = 0
    leased_sessions: int = 0
    starting_sessions: int = 0
    stale_status_files: int = 0
    reserved_ready_sessions: int = 0
    in_flight: int = 0
    pending_health: bool = False
    last_probe_at: float = 0.0
    last_success_at: float = 0.0
    last_capacity_check_at: float = 0.0
    quarantined_until: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    last_error: str | None = None
    last_capacity_error: str | None = None


@dataclass(frozen=True)
class RouteState:
    frontend_client_id: bytes
    frontend_request_id: bytes
    backend_index: int
    backend_address: str
    backend_request_id: bytes
    created_at: float
    reserved_capacity: bool = False


@dataclass(frozen=True)
class PendingFrontendRequest:
    frontend_client_id: bytes
    frontend_request_id: bytes
    method: bytes
    payload: bytes
    created_at: float


class ZMQRolloutGateway:
    """ZMQ request/reply load balancer for env-server replicas."""

    def __init__(
        self,
        *,
        bind_address: str,
        backend_addresses: list[str],
        backend_status_dirs: list[str],
        health_check_interval: float = 2.0,
        health_check_timeout: float = 5.0,
        request_timeout_s: float = 900.0,
        backend_quarantine_s: float = 30.0,
        capacity_check_interval: float = 5.0,
        capacity_wait_timeout_s: float | None = None,
        max_pending_requests: int = 0,
        status_stale_after_s: float = 120.0,
        request_id_factory: Callable[[], bytes] | None = None,
    ) -> None:
        if not backend_addresses:
            raise ValueError("backend_addresses must not be empty")
        if len(backend_status_dirs) != len(backend_addresses):
            raise ValueError("backend_status_dirs must match backend_addresses")
        if not all(backend_status_dirs):
            raise ValueError("backend_status_dirs entries must not be empty")
        self.bind_address = bind_address
        self.backends = [
            GatewayBackend(
                index=index, address=address, status_dir=backend_status_dirs[index]
            )
            for index, address in enumerate(backend_addresses)
        ]
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.request_timeout_s = request_timeout_s
        self.backend_quarantine_s = backend_quarantine_s
        self.capacity_check_interval = capacity_check_interval
        self.capacity_wait_timeout_s = (
            request_timeout_s
            if capacity_wait_timeout_s is None
            else capacity_wait_timeout_s
        )
        self.max_pending_requests = max(0, max_pending_requests)
        self.status_stale_after_s = status_stale_after_s
        self.request_id_factory = request_id_factory or (
            lambda: uuid.uuid4().hex.encode()
        )
        self.routes_by_frontend: dict[tuple[bytes, bytes], RouteState] = {}
        self.routes_by_backend: dict[tuple[int, bytes], RouteState] = {}
        self.pending_frontend_requests: OrderedDict[
            tuple[bytes, bytes], PendingFrontendRequest
        ] = OrderedDict()
        self.ctx: zmq.asyncio.Context | None = None
        self.frontend: Any | None = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def healthy_backend_count(self) -> int:
        now = time.monotonic()
        return sum(
            1 for backend in self.backends if self.backend_live(backend, now=now)
        )

    def start(self) -> None:
        """Bind the frontend ROUTER and connect cross-node backend DEALER sockets."""
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.bind_address)

        for backend in self.backends:
            socket = self.ctx.socket(zmq.DEALER)
            socket.setsockopt(zmq.SNDHWM, 0)
            socket.setsockopt(zmq.RCVHWM, 0)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
            socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)
            socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)
            socket.connect(backend.address)
            backend.socket = socket

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Serve frontend clients until the stop event is set."""
        if self.frontend is None:
            self.start()
        assert self.frontend is not None

        stop = stop_event or asyncio.Event()
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        for backend in self.backends:
            if backend.socket is not None:
                poller.register(backend.socket, zmq.POLLIN)

        self.logger.info(
            "ZMQ rollout gateway started on %s with %d backend(s)",
            self.bind_address,
            len(self.backends),
        )
        try:
            while not stop.is_set():
                await self.poll_backend_health()
                await self.expire_pending_requests()
                await self.expire_routes()
                await self.drain_pending_requests()
                events = dict(await poller.poll(timeout=100))
                if self.frontend in events:
                    frames = await self.frontend.recv_multipart()
                    await self.handle_frontend_message(frames)
                for backend in self.backends:
                    socket = backend.socket
                    if socket is None or socket not in events:
                        continue
                    frames = await socket.recv_multipart()
                    await self.handle_backend_message(backend, frames)
        except asyncio.CancelledError:
            pass
        finally:
            poller.unregister(self.frontend)
            for backend in self.backends:
                if backend.socket is not None:
                    poller.unregister(backend.socket)
            self.close()

    def close(self) -> None:
        if self.frontend is not None:
            self.frontend.close()
            self.frontend = None
        self.pending_frontend_requests.clear()
        for backend in self.backends:
            if backend.socket is not None:
                backend.socket.close()
                backend.socket = None
        if self.ctx is not None:
            self.ctx.term()
            self.ctx = None
        self.logger.info("ZMQ rollout gateway shut down")

    def select_backend(self) -> GatewayBackend | None:
        """Select the least-loaded healthy backend replica."""
        now = time.monotonic()
        candidates = [
            backend
            for backend in self.backends
            if self.backend_available(backend, now=now)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda backend: (
                backend_capacity_rank(backend),
                backend.in_flight,
                backend.index,
            ),
        )

    def backend_available(self, backend: GatewayBackend, *, now: float) -> bool:
        """Return whether a backend should receive rollout traffic."""
        return (
            backend.healthy
            and backend.capacity_ready
            and now >= backend.quarantined_until
        )

    def backend_live(self, backend: GatewayBackend, *, now: float) -> bool:
        """Return whether a backend should make frontend health checks pass."""
        return backend.healthy and now >= backend.quarantined_until

    async def handle_frontend_message(self, frames: list[bytes]) -> None:
        if len(frames) != 4:
            self.logger.warning(
                "Invalid frontend message: expected 4 frames, got %d", len(frames)
            )
            return
        client_id, request_id, method, payload = frames
        if method == HEALTH_METHOD:
            await self.send_frontend_response(
                client_id,
                request_id,
                health_response_bytes(self.healthy_backend_count > 0),
            )
            return
        await self.forward_request(client_id, request_id, payload, method=method)

    async def forward_request(
        self,
        frontend_client_id: bytes,
        frontend_request_id: bytes,
        payload: bytes,
        *,
        method: bytes = RUN_GROUP_METHOD,
    ) -> None:
        """Forward an opaque rollout payload or queue it until fleet capacity frees."""
        request = PendingFrontendRequest(
            frontend_client_id=frontend_client_id,
            frontend_request_id=frontend_request_id,
            method=method,
            payload=payload,
            created_at=time.monotonic(),
        )
        status = await self.try_dispatch_request(request)
        if status == "sent":
            return
        if status == "no_live_backend":
            await self.send_frontend_response(
                frontend_client_id,
                frontend_request_id,
                failed_response_bytes(NO_HEALTHY_BACKENDS_ERROR),
            )
            return

        key = (frontend_client_id, frontend_request_id)
        if (
            self.max_pending_requests
            and len(self.pending_frontend_requests) >= self.max_pending_requests
        ):
            await self.send_frontend_response(
                frontend_client_id,
                frontend_request_id,
                failed_response_bytes(PENDING_QUEUE_FULL_ERROR),
            )
            return
        self.pending_frontend_requests[key] = request

    async def try_dispatch_request(
        self,
        request: PendingFrontendRequest,
    ) -> DispatchStatus:
        attempted: set[int] = set()
        while True:
            backend = self.select_backend()
            if backend is None or backend.index in attempted:
                if self.healthy_backend_count <= 0:
                    return "no_live_backend"
                return "no_capacity"
            attempted.add(backend.index)
            backend_request_id = self.request_id_factory()
            reserved_capacity = self.reserve_backend_capacity(backend)
            route = RouteState(
                frontend_client_id=request.frontend_client_id,
                frontend_request_id=request.frontend_request_id,
                backend_index=backend.index,
                backend_address=backend.address,
                backend_request_id=backend_request_id,
                created_at=time.monotonic(),
                reserved_capacity=reserved_capacity,
            )
            self.routes_by_frontend[
                (request.frontend_client_id, request.frontend_request_id)
            ] = route
            self.routes_by_backend[(backend.index, backend_request_id)] = route
            backend.in_flight += 1
            try:
                await send_multipart(
                    backend.socket,
                    [backend_request_id, request.method, request.payload],
                )
                return "sent"
            except zmq.ZMQError as exc:
                self.drop_route(route)
                self.quarantine_backend(backend, str(exc))

    async def drain_pending_requests(self) -> None:
        """Route queued requests to whichever backend has free capacity now."""
        if not self.pending_frontend_requests:
            return
        self.refresh_all_backend_capacity(force=True)
        while self.pending_frontend_requests:
            key, request = next(iter(self.pending_frontend_requests.items()))
            status = await self.try_dispatch_request(request)
            if status == "sent":
                self.pending_frontend_requests.pop(key, None)
                continue
            if status == "no_live_backend":
                await self.fail_all_pending_requests(NO_HEALTHY_BACKENDS_ERROR)
            return

    async def fail_all_pending_requests(self, error: str) -> None:
        pending = list(self.pending_frontend_requests.values())
        self.pending_frontend_requests.clear()
        for request in pending:
            await self.send_frontend_response(
                request.frontend_client_id,
                request.frontend_request_id,
                failed_response_bytes(error),
            )

    async def handle_backend_message(
        self,
        backend: GatewayBackend,
        frames: list[bytes],
    ) -> None:
        """Forward a backend response to the original frontend request."""
        if len(frames) < 2:
            self.logger.warning(
                "Invalid backend message from %s: expected 2 frames, got %d",
                backend.address,
                len(frames),
            )
            return
        request_id, response_bytes = frames[0], frames[1]
        if request_id == HEALTH_REQUEST_ID:
            self.handle_health_response(backend, response_bytes)
            await self.drain_pending_requests()
            return
        route = self.routes_by_backend.get((backend.index, request_id))
        if route is None:
            self.logger.info(
                "Dropping late reply from %s for request %r: its route already "
                "expired (backend quarantined or the frontend request timed out)",
                backend.address,
                request_id,
            )
            return
        self.drop_route(route)
        await self.send_frontend_response(
            route.frontend_client_id,
            route.frontend_request_id,
            response_bytes,
        )
        await self.drain_pending_requests()

    async def send_frontend_response(
        self,
        client_id: bytes,
        request_id: bytes,
        response_bytes: bytes,
    ) -> None:
        if self.frontend is None:
            return
        try:
            await send_multipart(self.frontend, [client_id, request_id, response_bytes])
        except zmq.ZMQError as exc:
            self.logger.warning("Failed to send frontend response: %s", exc)

    def drop_route(self, route: RouteState) -> None:
        """Remove route bookkeeping and decrement backend load."""
        self.routes_by_frontend.pop(
            (route.frontend_client_id, route.frontend_request_id),
            None,
        )
        self.routes_by_backend.pop(
            (route.backend_index, route.backend_request_id), None
        )
        backend = self.backends[route.backend_index]
        backend.in_flight = max(0, backend.in_flight - 1)
        if route.reserved_capacity:
            self.release_backend_capacity_reservation(backend)

    async def expire_routes(self) -> None:
        """Fail frontend requests that are stuck behind an unresponsive backend."""
        if self.request_timeout_s <= 0:
            return
        now = time.monotonic()
        expired = [
            route
            for route in self.routes_by_frontend.values()
            if now - route.created_at >= self.request_timeout_s
        ]
        for route in expired:
            backend = self.backends[route.backend_index]
            self.drop_route(route)
            self.quarantine_backend(
                backend,
                f"{REQUEST_TIMEOUT_ERROR} ({self.request_timeout_s:.1f}s)",
                now=now,
            )
            await self.send_frontend_response(
                route.frontend_client_id,
                route.frontend_request_id,
                failed_response_bytes(
                    f"{REQUEST_TIMEOUT_ERROR}: {route.backend_address}"
                ),
            )

    async def expire_pending_requests(self) -> None:
        """Fail queued frontend requests that waited too long for capacity."""
        if self.capacity_wait_timeout_s <= 0:
            return
        now = time.monotonic()
        expired = [
            key
            for key, request in self.pending_frontend_requests.items()
            if now - request.created_at >= self.capacity_wait_timeout_s
        ]
        for key in expired:
            request = self.pending_frontend_requests.pop(key, None)
            if request is None:
                continue
            await self.send_frontend_response(
                request.frontend_client_id,
                request.frontend_request_id,
                failed_response_bytes(
                    f"{NO_BACKEND_CAPACITY_TIMEOUT_ERROR} "
                    f"({self.capacity_wait_timeout_s:.1f}s)"
                ),
            )

    def quarantine_backend(
        self,
        backend: GatewayBackend,
        error: str,
        *,
        now: float | None = None,
    ) -> None:
        """Temporarily remove a failing backend from routing consideration."""
        current = time.monotonic() if now is None else now
        backend.healthy = False
        backend.pending_health = False
        backend.quarantined_until = current + max(0.0, self.backend_quarantine_s)
        backend.last_error = error

    async def poll_backend_health(self) -> None:
        """Probe backend health using the env-server inline health protocol."""
        if self.health_check_interval <= 0:
            now = time.monotonic()
            for backend in self.backends:
                self.refresh_backend_capacity(backend, now=now)
                backend.healthy = True
            return

        now = time.monotonic()
        for backend in self.backends:
            self.refresh_backend_capacity(backend, now=now)
            if now < backend.quarantined_until:
                continue
            if (
                backend.pending_health
                and now - backend.last_probe_at > self.health_check_timeout
            ):
                backend.pending_health = False
                backend.healthy = False
                backend.last_error = "health check timed out"
            if backend.pending_health:
                continue
            if now - backend.last_probe_at < self.health_check_interval:
                continue
            backend.last_probe_at = now
            backend.pending_health = True
            try:
                await send_multipart(
                    backend.socket,
                    [HEALTH_REQUEST_ID, HEALTH_METHOD, HEALTH_REQUEST_PAYLOAD],
                )
            except zmq.ZMQError as exc:
                backend.pending_health = False
                backend.healthy = False
                backend.last_error = str(exc)

    def handle_health_response(
        self,
        backend: GatewayBackend,
        response_bytes: bytes,
    ) -> None:
        backend.pending_health = False
        try:
            response = msgpack.unpackb(response_bytes, raw=False)
        except Exception as exc:
            backend.healthy = False
            backend.last_error = f"invalid health response: {exc}"
            return
        backend.healthy = bool(response.get("success"))
        backend.last_success_at = time.monotonic() if backend.healthy else 0.0
        backend.last_error = response.get("error")

    def refresh_backend_capacity(
        self,
        backend: GatewayBackend,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        """Refresh desktop-pool capacity from the backend status directory."""
        assert backend.status_dir, (
            "GatewayBackend.status_dir is required (enforced in "
            "ZMQRolloutGateway.__init__); a gateway backend with no status dir "
            "would report capacity_ready unconditionally and route to it blind"
        )
        current = time.monotonic() if now is None else now
        if (
            not force
            and current - backend.last_capacity_check_at < self.capacity_check_interval
        ):
            return
        backend.last_capacity_check_at = current

        status_dir = Path(backend.status_dir)
        statuses = read_statuses(status_dir, recursive=False)
        if not statuses:
            backend.ready_sessions = 0
            backend.leased_sessions = 0
            backend.starting_sessions = 0
            backend.stale_status_files = 0
            backend.reserved_ready_sessions = 0
            backend.capacity_ready = False
            backend.last_capacity_error = (
                f"no desktop pool status files in {status_dir}"
            )
            return

        status_now = time.time()
        active_statuses = active_worker_statuses(
            statuses,
            now=status_now,
            stale_after_s=self.status_stale_after_s,
        )
        backend.stale_status_files = len(
            stale_worker_statuses(
                statuses,
                now=status_now,
                stale_after_s=self.status_stale_after_s,
            )
        )
        backend.ready_sessions = sum_int_field(active_statuses, "ready")
        backend.leased_sessions = sum_int_field(active_statuses, "leased")
        backend.starting_sessions = sum_int_field(active_statuses, "starting")
        backend.reserved_ready_sessions = min(
            backend.reserved_ready_sessions,
            backend.ready_sessions,
        )
        backend.capacity_ready = available_ready_sessions(backend) > 0
        backend.last_capacity_error = (
            None
            if backend.capacity_ready
            else f"no ready desktop sessions in {status_dir}"
        )

    def refresh_all_backend_capacity(self, *, force: bool = False) -> None:
        now = time.monotonic()
        for backend in self.backends:
            self.refresh_backend_capacity(backend, now=now, force=force)

    def reserve_backend_capacity(self, backend: GatewayBackend) -> bool:
        """Reserve one observed ready desktop slot after routing a request."""
        assert backend.status_dir, (
            "GatewayBackend.status_dir is required (enforced in "
            "ZMQRolloutGateway.__init__)"
        )
        if available_ready_sessions(backend) <= 0:
            return False
        backend.reserved_ready_sessions = min(
            backend.ready_sessions,
            backend.reserved_ready_sessions + 1,
        )
        backend.capacity_ready = available_ready_sessions(backend) > 0
        backend.last_capacity_error = (
            None
            if backend.capacity_ready
            else f"all observed ready desktop sessions are reserved for {backend.status_dir}"
        )
        return True

    def release_backend_capacity_reservation(self, backend: GatewayBackend) -> None:
        """Release one optimistic ready-slot reservation for a completed route."""
        if backend.reserved_ready_sessions <= 0:
            return
        backend.reserved_ready_sessions -= 1
        backend.capacity_ready = available_ready_sessions(backend) > 0
        if backend.capacity_ready:
            backend.last_capacity_error = None


async def send_multipart(socket: Any | None, frames: list[bytes]) -> None:
    if socket is None:
        raise zmq.ZMQError(zmq.ENOTSOCK)
    await socket.send_multipart(frames)


def failed_response_bytes(error: str) -> bytes:
    return msgpack.packb(
        {"success": False, "error": error},
        use_bin_type=True,
    )


def health_response_bytes(success: bool) -> bytes:
    return msgpack.packb(
        {
            "success": success,
            "error": None if success else NO_HEALTHY_BACKENDS_ERROR,
        },
        use_bin_type=True,
    )


def available_ready_sessions(backend: GatewayBackend) -> int:
    """VM-pool capacity read from status files, minus outstanding reservations."""
    return max(0, backend.ready_sessions - backend.reserved_ready_sessions)


def backend_capacity_rank(backend: GatewayBackend) -> int:
    """Rank 0 when a leased machine is free on this backend, 1 otherwise."""
    if available_ready_sessions(backend) > 0:
        return 0
    return 1


GATEWAY_TUNABLES = (
    "health_check_interval",
    "health_check_timeout",
    "request_timeout_s",
    "backend_quarantine_s",
    "capacity_check_interval",
    "capacity_wait_timeout_s",
    "max_pending_requests",
    "status_stale_after_s",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = wait_for_registry_gateway(args)
    logging.info(
        "Starting rollout gateway on %s for %d backend(s)",
        config["bind_address"],
        len(config["backend_addresses"]),
    )
    gateway = ZMQRolloutGateway(
        bind_address=config["bind_address"],
        backend_addresses=config["backend_addresses"],
        backend_status_dirs=config["backend_status_dirs"],
        **{name: getattr(args, name) for name in GATEWAY_TUNABLES},
    )
    asyncio.run(gateway.serve())
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_runtime_env_file()
    layout = FleetRunLayout.from_env(os.environ)
    parser = argparse.ArgumentParser(
        description="Run a cross-node ZMQ rollout broker for env-server replicas."
    )
    parser.add_argument("--registry", type=Path, default=layout.registry_path)
    parser.add_argument("--bind-address")
    parser.add_argument("--backend-address", action="append", default=[])
    parser.add_argument("--backend-status-dir", action="append", default=[])
    parser.add_argument("--wait-timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("--health-check-timeout", type=float, default=5.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--backend-quarantine-s", type=float, default=30.0)
    parser.add_argument("--capacity-check-interval", type=float, default=5.0)
    parser.add_argument("--capacity-wait-timeout-s", type=float)
    parser.add_argument("--max-pending-requests", type=int, default=0)
    parser.add_argument("--status-stale-after-s", type=float, default=120.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def wait_for_registry_gateway(args: argparse.Namespace) -> dict[str, Any]:
    """Wait for the fleet registry, honoring --bind-address/--backend-address overrides."""
    deadline = time.monotonic() + args.wait_timeout_s
    last_error = "registry not read yet"
    while time.monotonic() <= deadline:
        try:
            registry = read_registry(args.registry)
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(args.poll_s)
            continue
        metadata = registry.metadata
        expected = int(metadata.get("expected_env_servers", 0) or 0)
        backend_addresses = list(args.backend_address) or gateway_backend_addresses(
            metadata,
            registry.servers,
        )
        backend_status_dirs = list(
            args.backend_status_dir
        ) or gateway_backend_status_dirs(
            registry.servers,
            backend_addresses,
        )
        bind_address = args.bind_address or gateway_bind_address(metadata)
        if bind_address and backend_addresses and len(registry.servers) >= expected:
            # A backend with no status dir reports capacity_ready unconditionally,
            # so the broker would route to it without ever reading its pool. Every
            # node has registered by here, so an unmatched address means the
            # gateway's backend_addresses (built from slurm_node_addrs on rank 0)
            # and server.public_address (built from scontrol NodeAddr on each node)
            # disagree about a host string, and the exact-match join in
            # gateway_backend_status_dirs silently missed.
            unmatched = [
                address
                for address, status_dir in zip(
                    backend_addresses, backend_status_dirs, strict=True
                )
                if status_dir is None
            ]
            if unmatched:
                raise ValueError(
                    f"no registered env server publishes {unmatched}; the gateway "
                    "cannot read their desktop-pool capacity and would route to "
                    "them blind. Registered addresses: "
                    f"{[server.public_address for server in registry.servers]}"
                )
            return {
                "bind_address": bind_address,
                "backend_addresses": backend_addresses,
                "backend_status_dirs": backend_status_dirs,
            }
        last_error = (
            f"bind={bool(bind_address)} backends={len(backend_addresses)} "
            f"registered={len(registry.servers)} expected={expected}"
        )
        time.sleep(args.poll_s)
    raise TimeoutError(f"gateway registry metadata was not ready: {last_error}")


def gateway_bind_address(metadata: Mapping[str, Any]) -> str | None:
    gateway = metadata.get("gateway")
    if not isinstance(gateway, Mapping):
        return None
    address = gateway.get("bind_address")
    return str(address) if address else None


def gateway_backend_addresses(
    metadata: Mapping[str, Any],
    servers: list[EnvServerSpec],
) -> list[str]:
    gateway = metadata.get("gateway")
    if isinstance(gateway, Mapping):
        addresses = gateway.get("backend_addresses")
        if isinstance(addresses, list):
            return [str(address) for address in addresses if address]
    return [server.public_address for server in servers]


def gateway_backend_status_dirs(
    servers: list[EnvServerSpec],
    backend_addresses: list[str],
) -> list[str | None]:
    """Return pool status dirs ordered like gateway backend addresses."""
    by_address = {server.public_address: server.pool_status_dir for server in servers}
    return [by_address.get(address) for address in backend_addresses]


if __name__ == "__main__":
    raise SystemExit(main())
