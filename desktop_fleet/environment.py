from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

ENVIRONMENT_CONTRACT_VERSION = 2
ENVIRONMENT_CONTRACT_ENV = "ENV_FLEET_ENVIRONMENT_CONTRACT"
_SOURCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_LEGACY_METADATA_KEYS = frozenset(
    {
        "env_id",
        "env_name_prefix",
        "harness",
        "max_tasks",
        "shuffle_seed",
        "task_base_path",
    }
)


@dataclass(frozen=True)
class EnvironmentSession:
    taskset: Mapping[str, Any]
    harness: Mapping[str, Any]
    output_dir: Path
    rollout_timeout: float
    max_retries: int
    max_turns: int
    status_dir_path: tuple[str, ...]

    def __post_init__(self) -> None:
        taskset = _config_mapping(self.taskset, name="session.taskset")
        harness = _config_mapping(self.harness, name="session.harness")
        output_dir = Path(self.output_dir)
        if not output_dir.is_absolute():
            raise ValueError("session.output_dir must be an absolute path")
        _utf8(str(output_dir), name="session.output_dir")
        rollout_timeout = _positive_number(
            self.rollout_timeout, name="session.rollout_timeout"
        )
        _integer(self.max_retries, name="session.max_retries", minimum=0)
        _integer(self.max_turns, name="session.max_turns", minimum=1)
        status_dir_path = _config_path(
            self.status_dir_path, name="session.status_dir_path"
        )
        _require_injection_target(harness, status_dir_path)
        _integer(
            _path_parent(harness, status_dir_path).get("min_ready_sessions"),
            name="session harness min_ready_sessions",
            minimum=1,
        )
        object.__setattr__(self, "taskset", taskset)
        object.__setattr__(self, "harness", harness)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "rollout_timeout", rollout_timeout)
        object.__setattr__(self, "status_dir_path", status_dir_path)

    @property
    def min_ready_sessions(self) -> int:
        return _integer(
            _path_parent(self.harness, self.status_dir_path).get(
                "min_ready_sessions"
            ),
            name="session harness min_ready_sessions",
            minimum=1,
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "taskset": deepcopy(dict(self.taskset)),
            "harness": deepcopy(dict(self.harness)),
            "output_dir": str(self.output_dir),
            "rollout_timeout": self.rollout_timeout,
            "max_retries": self.max_retries,
            "max_turns": self.max_turns,
            "status_dir_path": list(self.status_dir_path),
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {
                "taskset",
                "harness",
                "output_dir",
                "rollout_timeout",
                "max_retries",
                "max_turns",
                "status_dir_path",
            },
            name="environment session",
        )
        return cls(
            taskset=_mapping(payload, "taskset", name="environment session"),
            harness=_mapping(payload, "harness", name="environment session"),
            output_dir=Path(_string(payload, "output_dir", name="environment session")),
            rollout_timeout=_number(
                payload, "rollout_timeout", name="environment session"
            ),
            max_retries=_int(payload, "max_retries", name="environment session"),
            max_turns=_int(payload, "max_turns", name="environment session"),
            status_dir_path=_serialized_path(
                payload, "status_dir_path", name="environment session"
            ),
        )


@dataclass(frozen=True)
class EnvironmentSource:
    name: str
    taskset: Mapping[str, Any]
    harness: Mapping[str, Any]

    def __post_init__(self) -> None:
        name = self.name if isinstance(self.name, str) else ""
        if not _SOURCE_NAME.fullmatch(name):
            raise ValueError(
                "source.name must be a 1-64 character identifier containing only "
                "letters, digits, underscores, and hyphens"
            )
        taskset = _config_mapping(self.taskset, name="source.taskset")
        harness = _config_mapping(self.harness, name="source.harness")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "taskset", taskset)
        object.__setattr__(self, "harness", harness)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "taskset": deepcopy(dict(self.taskset)),
            "harness": deepcopy(dict(self.harness)),
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {"name", "taskset", "harness"},
            name="environment source",
        )
        return cls(
            name=_string(payload, "name", name="environment source"),
            taskset=_mapping(payload, "taskset", name="environment source"),
            harness=_mapping(payload, "harness", name="environment source"),
        )


@dataclass(frozen=True)
class EnvironmentContract:
    session: EnvironmentSession
    source: EnvironmentSource

    def as_metadata(self) -> dict[str, Any]:
        return {
            "version": ENVIRONMENT_CONTRACT_VERSION,
            "session": self.session.as_metadata(),
            "source": self.source.as_metadata(),
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {"version", "session", "source"},
            name="environment contract",
        )
        version = payload["version"]
        if type(version) is not int or version != ENVIRONMENT_CONTRACT_VERSION:
            raise ValueError(f"unsupported environment contract version: {version!r}")
        return cls(
            session=EnvironmentSession.from_metadata(
                _mapping(payload, "session", name="environment contract")
            ),
            source=EnvironmentSource.from_metadata(
                _mapping(payload, "source", name="environment contract")
            ),
        )


def read_environment_contract(path: str | Path) -> EnvironmentContract:
    resolved = Path(path)
    if not resolved.is_absolute():
        raise ValueError("environment contract path must be absolute")
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("environment contract must be a JSON object")
    return EnvironmentContract.from_metadata(payload)


def environment_contract_from_registry_metadata(
    metadata: Mapping[str, Any],
) -> EnvironmentContract:
    legacy = sorted(_LEGACY_METADATA_KEYS & metadata.keys())
    if legacy:
        raise ValueError(
            f"legacy flat environment metadata is unsupported: {', '.join(legacy)}"
        )
    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("registry metadata is missing environment contract")
    return EnvironmentContract.from_metadata(environment)


def inject_status_dir(
    harness: Mapping[str, Any],
    path: Sequence[str],
    status_dir: str,
) -> dict[str, Any]:
    rendered = deepcopy(dict(harness))
    resolved_path = _config_path(path, name="session.status_dir_path")
    parent = _path_parent(rendered, resolved_path)
    if resolved_path[-1] in parent:
        raise ValueError(
            f"session.status_dir_path target already exists: {'.'.join(resolved_path)}"
        )
    parent[resolved_path[-1]] = status_dir
    return rendered


def _require_injection_target(
    harness: Mapping[str, Any], path: tuple[str, ...]
) -> None:
    parent = _path_parent(harness, path)
    if path[-1] in parent:
        raise ValueError(f"session.status_dir_path target already exists: {'.'.join(path)}")


def _path_parent(
    config: Mapping[str, Any], path: Sequence[str]
) -> dict[str, Any]:
    current: Mapping[str, Any] = config
    for part in path[:-1]:
        value = current.get(part)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"config path parent does not exist as a table: {'.'.join(path)}"
            )
        current = value
    if not isinstance(current, dict):
        raise ValueError(f"config path parent is immutable: {'.'.join(path)}")
    return current


def _config_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a table")
    copied = _config_value(value, name=name)
    assert isinstance(copied, dict)
    return copied


def _config_value(value: object, *, name: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be non-empty strings")
            _utf8(key, name=f"{name} key")
            result[key] = _config_value(item, name=name)
        return result
    if isinstance(value, list | tuple):
        return [_config_value(item, name=name) for item in value]
    if isinstance(value, str):
        return _utf8(value, name=f"{name} string")
    if type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{name} contains a value unsupported by fleet TOML: {value!r}")


def _config_path(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list | tuple)
        or not value
        or not all(isinstance(part, str) and part for part in value)
    ):
        raise ValueError(f"{name} must contain non-empty string keys")
    for part in value:
        _utf8(part, name=name)
    return tuple(value)


def _serialized_config_path(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return _config_path(value, name=name)


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{name} keys do not match the contract; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _mapping(payload: Mapping[str, Any], key: str, *, name: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} {key} must be a table")
    return value


def _string(payload: Mapping[str, Any], key: str, *, name: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} {key} must be a non-empty string")
    return _utf8(value, name=f"{name} {key}")


def _utf8(value: str, *, name: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    return value


def _int(payload: Mapping[str, Any], key: str, *, name: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ValueError(f"{name} {key} must be an integer")
    return value


def _number(payload: Mapping[str, Any], key: str, *, name: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} {key} must be a number")
    return float(value)


def _serialized_path(
    payload: Mapping[str, Any], key: str, *, name: str
) -> tuple[str, ...]:
    return _serialized_config_path(payload.get(key), name=f"{name} {key}")


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def _integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value
