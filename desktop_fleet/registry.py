"""The fleet registry: the durable, lock-protected list of live machines.

Every node in the allocation upserts its own replicas into one shared JSON file,
so a consumer that starts later can discover the whole fleet from disk alone.
"""

from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, overload

from desktop_fleet.environment import environment_contract_from_registry_metadata
from desktop_fleet.spec import EnvServerSpec

_AGGREGATED_METADATA_KEYS = frozenset({"node_services"})


@dataclass
class EnvFleetRegistry:
    run_id: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    servers: list[EnvServerSpec] = field(default_factory=list)

    @classmethod
    def empty(cls, *, run_id: str, metadata: Mapping[str, Any]) -> Self:
        _validate_registry_metadata(metadata)
        now = time.time()
        return cls(
            run_id=run_id,
            created_at=now,
            updated_at=now,
            metadata=deepcopy(dict(metadata)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        metadata = dict(payload.get("metadata") or {})
        _validate_registry_metadata(metadata)
        servers = [
            EnvServerSpec(**server)
            for server in payload.get("servers", [])
            if isinstance(server, Mapping)
        ]
        return cls(
            run_id=str(payload["run_id"]),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
            metadata=metadata,
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
    _validate_registry_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if path.exists():
            registry = read_registry(path)
            if registry.run_id != run_id:
                raise ValueError(
                    f"registry run_id is immutable: {registry.run_id!r} != {run_id!r}"
                )
            _merge_registry_metadata(registry.metadata, metadata)
        else:
            registry = EnvFleetRegistry.empty(run_id=run_id, metadata=metadata)

        by_name = {server.name: server for server in registry.servers}
        for server in servers:
            by_name[server.name] = server
        registry.servers = sorted(by_name.values(), key=lambda server: server.name)
        registry.updated_at = time.time()
        _write_registry(path, registry)
        return registry


def _merge_registry_metadata(
    target: dict[str, Any], values: Mapping[str, Any]
) -> None:
    fixed_target = {
        key: value for key, value in target.items() if key not in _AGGREGATED_METADATA_KEYS
    }
    fixed_values = {
        key: value for key, value in values.items() if key not in _AGGREGATED_METADATA_KEYS
    }
    if _json_identity(fixed_target) != _json_identity(fixed_values):
        differing = sorted(
            key
            for key in fixed_target.keys() | fixed_values.keys()
            if key not in fixed_target
            or key not in fixed_values
            or _json_identity(fixed_target[key]) != _json_identity(fixed_values[key])
        )
        raise ValueError(
            f"registry shared metadata is immutable; conflicting keys: {differing}"
        )
    for key in _AGGREGATED_METADATA_KEYS:
        if key not in values:
            continue
        incoming = values[key]
        if not isinstance(incoming, Mapping):
            raise ValueError(f"registry aggregated metadata {key} must be a table")
        current = target.setdefault(key, {})
        if not isinstance(current, dict):
            raise ValueError(f"registry aggregated metadata {key} must be a table")
        _merge_aggregated_metadata(current, incoming, path=(key,))


def _validate_registry_metadata(metadata: Mapping[str, Any]) -> None:
    environment_contract_from_registry_metadata(metadata)
    try:
        _json_identity(metadata)
    except (TypeError, ValueError) as error:
        raise ValueError("registry metadata must be finite JSON") from error
    for key in _AGGREGATED_METADATA_KEYS:
        if key in metadata and not isinstance(metadata[key], Mapping):
            raise ValueError(f"registry aggregated metadata {key} must be a table")


def _merge_aggregated_metadata(
    target: dict[str, Any],
    values: Mapping[str, Any],
    *,
    path: tuple[str, ...],
) -> None:
    for key, value in values.items():
        if key not in target:
            target[key] = deepcopy(value)
            continue
        current = target[key]
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_aggregated_metadata(current, value, path=(*path, key))
            continue
        if _json_identity(current) != _json_identity(value):
            location = ".".join((*path, key))
            raise ValueError(
                f"registry aggregated metadata writer conflict at {location}"
            )


def _json_identity(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_registry(path: Path, registry: EnvFleetRegistry) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(registry.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
