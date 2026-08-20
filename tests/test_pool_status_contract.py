"""The desktop-pool status file, checked across the two packages that share it.

``desktop`` writes it; this package derives routing capacity and writer liveness
from it.  Neither side's own suite can see the other, and the consumer reads a
missing field as zero rather than raising, so a field renamed on one side alone
leaves both suites green and the routing wrong -- ``backend_capacity_rank``
answering from counters that are all zero.  This is the only test that spans the
boundary, and it is the reason the two packages live in one repository.

The producer is a real ``DesktopSessionPool`` writing a real status file.  Only
the port allocator and the pooled session are doubled: they are the two seams the
pool injects, and neither is part of the contract under test.  What the pool
wrote is then copied to a directory the pool does not own, because the pool
keeps rewriting its own file from a heartbeat thread.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pytest
from desktop.vm.pool import (
    CONSUMED_STATUS_FIELDS,
    DesktopPoolConfig,
    DesktopSessionPool,
    PortLease,
    ports_for_worker,
)

from desktop_fleet.broker import (
    GatewayBackend,
    available_ready_sessions,
    backend_capacity_rank,
)
from desktop_fleet.readiness import (
    active_worker_statuses,
    read_statuses,
    stale_worker_statuses,
    sum_int_field,
)
from desktop_fleet.supervise import read_pool_health

READY_SESSIONS = 3
STALE_AFTER_S = 120.0

#: The status fields this package reads with *no* default, so a rename raises
#: instead of reading as zero. ``desktop.vm.pool.CONSUMED_STATUS_FIELDS`` covers
#: only the ``.get(field, 0)`` ones and names neither of these, which is why they
#: are asserted here.
REQUIRED_STATUS_FIELDS = ("startup_timeout_s", "starting_sessions")


class _PooledSession:
    """The pool's entire requirement of a session is ``close()``."""

    def __init__(self, lease: PortLease) -> None:
        self.lease = lease

    def close(self) -> None:
        pass


def _port_allocator():
    slots = itertools.count()

    def allocate(*, lock_dir: Path, log_dir, work_dir=None) -> PortLease:
        slot = next(slots)
        workdir = Path(work_dir or lock_dir) / f"w{slot}"
        logdir = Path(log_dir) / f"w{slot}"
        workdir.mkdir(parents=True, exist_ok=True)
        logdir.mkdir(parents=True, exist_ok=True)
        return PortLease(
            ports=ports_for_worker(51000, slot),
            slot=slot,
            workdir=workdir,
            logdir=logdir,
            _lock_file=(lock_dir / f"w{slot}.lock").open("w"),
        )

    return allocate


@pytest.fixture
def produced(tmp_path) -> tuple[dict, Path]:
    """One real status payload, and a directory holding only that payload."""
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=READY_SESSIONS, max_sessions=READY_SESSIONS
        ),
        root_dir=tmp_path / "pool",
        session_factory=_PooledSession,
        port_allocator=_port_allocator(),
    )
    try:
        pool.start()
        payload = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if pool.status_path.is_file():
                payload = json.loads(pool.status_path.read_text())
                if payload["ready"] == READY_SESSIONS:
                    break
            time.sleep(0.01)
        assert payload is not None and payload["ready"] == READY_SESSIONS, (
            f"the pool never reported {READY_SESSIONS} ready sessions: {payload}"
        )
        name = pool.status_path.name
    finally:
        pool.close()

    status_dir = tmp_path / "observed"
    status_dir.mkdir()
    (status_dir / name).write_text(json.dumps(payload))
    return payload, status_dir


def _active(status_dir: Path, *, now: float):
    return active_worker_statuses(
        read_statuses(status_dir, recursive=False), now=now, stale_after_s=STALE_AFTER_S
    )


def test_the_producer_emits_every_field_this_package_consumes(produced):
    payload, _ = produced
    missing = [name for name in CONSUMED_STATUS_FIELDS if name not in payload]
    assert missing == [], f"desktop stopped writing fields desktop_fleet reads: {missing}"


def test_the_consumer_derives_the_producers_own_counts(produced):
    """Presence is not agreement: the numbers have to survive the round trip."""
    payload, status_dir = produced
    active = _active(status_dir, now=payload["updated_at"])

    assert len(active) == 1
    for name in ("ready", "starting", "leased"):
        assert sum_int_field(active, name) == payload[name], name
    assert sum_int_field(active, "ready") == READY_SESSIONS


def test_a_backend_is_ranked_from_the_counts_the_pool_actually_wrote(produced):
    """The routing decision itself, not the counter behind it.

    A renamed capacity field ranks every backend equal, and a fleet that cannot
    tell a free machine from a busy one queues nothing and routes blind.
    """
    payload, status_dir = produced
    backend = GatewayBackend(
        index=0, address="tcp://127.0.0.1:1", status_dir=str(status_dir)
    )
    backend.ready_sessions = sum_int_field(
        _active(status_dir, now=payload["updated_at"]), "ready"
    )

    assert available_ready_sessions(backend) == READY_SESSIONS
    assert backend_capacity_rank(backend) == 0
    backend.reserved_ready_sessions = READY_SESSIONS
    assert backend_capacity_rank(backend) == 1


def test_the_startup_budget_the_supervisor_scores_stuck_desktops_against(produced):
    """The restart decision, over the payload the pool actually wrote.

    ``summarize_starting_sessions`` divides the pool's starting desktops into
    ones inside their startup budget and ones past it, and only the latter
    justify a restart. It reads ``startup_timeout_s`` and each
    ``starting_sessions[].created_at`` with no default on purpose: scoring an
    unreadable session as *within* budget is the one answer that is never safe,
    because it is what stops a replica whose desktops are permanently stuck
    starting from ever being restarted.
    """
    payload, status_dir = produced

    missing = [name for name in REQUIRED_STATUS_FIELDS if name not in payload]
    assert missing == [], f"desktop stopped writing fields desktop_fleet requires: {missing}"
    assert isinstance(payload["starting_sessions"], list)
    assert all("created_at" in session for session in payload["starting_sessions"])

    health = read_pool_health(status_dir, status_stale_after_s=STALE_AFTER_S)

    assert health.starting == payload["starting"]
    assert health.fresh_starting + health.stale_starting == len(
        payload["starting_sessions"]
    )


@pytest.mark.parametrize("field", ["closed", "updated_at"])
def test_a_liveness_field_the_producer_writes_still_disqualifies_a_worker(
    produced, field
):
    """``closed`` and ``updated_at`` are how a dead writer stops being counted.

    Rename either on the producer and a shut-down or hung pool goes on
    advertising its last-known ready count for as long as the file exists.
    """
    payload, status_dir = produced
    now = payload["updated_at"]
    dead = dict(payload)
    dead[field] = True if field == "closed" else now - 10 * STALE_AFTER_S
    next(status_dir.glob("*.json")).write_text(json.dumps(dead))

    active = _active(status_dir, now=now)
    assert active == []
    assert sum_int_field(active, "ready") == 0

    statuses = read_statuses(status_dir, recursive=False)
    stale = stale_worker_statuses(statuses, now=now, stale_after_s=STALE_AFTER_S)
    # A closed pool is not stale, it is finished; only the hung one is stale.
    assert len(stale) == (0 if field == "closed" else 1)
