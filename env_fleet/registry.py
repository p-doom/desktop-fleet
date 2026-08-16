"""The fleet registry: the durable, lock-protected list of live machines.

Every node in the allocation upserts its own replicas into one shared JSON file,
so a consumer that starts later can discover the whole fleet from disk alone.
"""

from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, overload

from env_fleet.spec import EnvServerSpec


@dataclass
class EnvFleetRegistry:
    run_id: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    servers: list[EnvServerSpec] = field(default_factory=list)

    @classmethod
    def empty(cls, *, run_id: str, metadata: Mapping[str, Any]) -> Self:
        now = time.time()
        return cls(
            run_id=run_id,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        servers = [
            EnvServerSpec(**server)
            for server in payload.get("servers", [])
            if isinstance(server, Mapping)
        ]
        return cls(
            run_id=str(payload["run_id"]),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
            metadata=dict(payload.get("metadata") or {}),
            servers=servers,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "servers": [asdict(server) for server in self.servers],
        }


@overload
def read_registry(
    path: Path, *, if_missing: Literal["raise"] = "raise"
) -> EnvFleetRegistry: ...


@overload
def read_registry(
    path: Path, *, if_missing: Literal["none"]
) -> EnvFleetRegistry | None: ...


def read_registry(
    path: Path,
    *,
    if_missing: Literal["raise", "none"] = "raise",
) -> EnvFleetRegistry | None:
    """Read the fleet registry, always raising on a corrupt file.

    ``if_missing`` selects only how a missing file is handled: ``"raise"``
    (the default) lets ``FileNotFoundError`` propagate; ``"none"`` returns
    ``None`` instead, for callers that treat "fleet not started yet" as a
    normal, expected state rather than an error.
    """
    if if_missing == "none" and not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"registry is not a JSON object: {path}")
    return EnvFleetRegistry.from_dict(payload)


def read_registry_if_ready(path: Path) -> tuple[EnvFleetRegistry | None, str | None]:
    """Read the fleet registry and return an error string instead of raising."""
    if not path.exists():
        return None, "missing"
    try:
        return read_registry(path), None
    except Exception as exc:
        return None, repr(exc)


def upsert_registry(
    *,
    path: Path,
    run_id: str,
    metadata: Mapping[str, Any],
    servers: Iterable[EnvServerSpec],
) -> EnvFleetRegistry:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if path.exists():
            registry = read_registry(path)
            registry.metadata.update(dict(metadata))
        else:
            registry = EnvFleetRegistry.empty(run_id=run_id, metadata=metadata)

        by_name = {server.name: server for server in registry.servers}
        for server in servers:
            by_name[server.name] = server
        registry.servers = sorted(by_name.values(), key=lambda server: server.name)
        registry.updated_at = time.time()
        _write_registry(path, registry)
        return registry


def _write_registry(path: Path, registry: EnvFleetRegistry) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(registry.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
