"""Machine specs, run layout, and verifiers env-server config rendering.

desktop-fleet manages *machines*. This module owns the three things a machine has:
an identity (:class:`EnvServerSpec`), a place on disk (:class:`FleetRunLayout`),
and a rendered ``verifiers`` env-server config. Nothing here knows about
actions, grammars, or any particular trainer -- trainer-specific paths ride
along opaquely in :attr:`FleetRunLayout.consumer_paths`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Self

# Keep the shared desktop-pool path component short as a fallback; QEMU AF_UNIX
# socket paths should use DesktopPoolConfig.runtime_dir when available.
DEFAULT_DESKTOP_POOL_DIR = "pool"

# ``name=/abs/path[,name=/abs/path]`` -- how a consumer injects its own paths
# into the layout without desktop-fleet knowing what they mean.
CONSUMER_PATHS_ENV = "ENV_FLEET_CONSUMER_PATHS"


def require_absolute_path(value: str | Path, *, name: str = "path") -> Path:
    raw_value = str(value)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(
            f"{name} must be an absolute path; got {raw_value!r}. "
            "Do not use '~', relative paths, or unexpanded shell variables."
        )
    return path


def project_root(env: Mapping[str, str] = os.environ) -> Path:
    """Return the operator's project checkout, used to resolve helper scripts."""
    value = env.get("ENV_FLEET_PROJECT_ROOT")
    if value:
        return require_absolute_path(value, name="ENV_FLEET_PROJECT_ROOT")
    return Path.cwd()


def scratch_root(env: Mapping[str, str] = os.environ) -> Path:
    scratch = env.get("SCRATCH")
    if scratch:
        return require_absolute_path(scratch, name="SCRATCH")
    return project_root(env) / ".scratch"


def scratch_subdir(*parts: str, env: Mapping[str, str] = os.environ) -> Path:
    return scratch_root(env).joinpath(*parts)


def slurm_run_id(env: Mapping[str, str] = os.environ) -> str:
    """The run id to *look under* when no command line named one.

    Only the read-only commands may take this fallback. ``supervise prepare``
    creates the run's directories and writes its registry, so it requires an
    explicit id: two off-scheduler prepares sharing one would sum each other's
    ready counts through one status tree.
    """
    return env.get("SLURM_JOB_ID") or "manual"


def load_runtime_env_file(
    env: MutableMapping[str, str] = os.environ,
    *,
    path: str | Path | None = None,
) -> Path | None:
    """Load ``KEY=value`` runtime defaults into ``env`` before argument parsing.

    A variable already present in ``env`` wins.  This runs before argparse, so
    every ``Opt`` with an environment name reads it: overwriting would make
    ``OSWORLD_FLEET_BASE_PORT=5300 python -m desktop_fleet.supervise ...`` lose
    silently to whatever the file happens to say.
    """
    resolved = _runtime_env_file_path(env=env, path=path)
    if resolved is None or not resolved.is_file():
        return None

    for key, value in _runtime_env_values(resolved).items():
        env.setdefault(key, value)
    return resolved


def _runtime_env_file_path(
    env: MutableMapping[str, str] = os.environ,
    path: str | Path | None = None,
) -> Path | None:
    if path is not None:
        return require_absolute_path(path, name="runtime env file path")
    explicit = env.get("RL_RUNTIME_ENV_FILE")
    if explicit is not None:
        if not explicit:
            return None
        return require_absolute_path(explicit, name="RL_RUNTIME_ENV_FILE")
    return project_root(env) / ".env"


def _runtime_env_values(path: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    values: dict[str, str] = {}
    for key, value in dotenv_values(path).items():
        if value is None:
            raise ValueError(f"{path}: expected KEY=value for {key!r}")
        values[key] = value
    return values


@dataclass(frozen=True)
class EnvServerSpec:
    name: str
    host: str
    port: int
    bind_address: str
    public_address: str
    num_workers: int
    node_rank: int
    local_index: int
    replica_index: int
    replica_count: int
    config_path: str
    log_path: str
    pool_status_dir: str | None = None


def default_public_host(
    env: Mapping[str, str] = os.environ,
    *,
    run_command: Callable[[list[str]], str] | None = None,
    fqdn_func: Callable[[], str] = socket.getfqdn,
    hostname_func: Callable[[], str] = socket.gethostname,
) -> str:
    if explicit_host := env.get("OSWORLD_FLEET_HOST"):
        return explicit_host

    if slurm_host := _slurm_node_address(
        env,
        run_command=run_command,
        hostname_func=hostname_func,
    ):
        return slurm_host

    return fqdn_func() or hostname_func()


def make_server_specs(
    *,
    host: str,
    bind_host: str,
    base_port: int,
    node_rank: int,
    servers_per_node: int,
    workers_per_server: int,
    replica_count: int,
    replica_offset: int,
    name_prefix: str,
    config_dir: Path,
    log_dir: Path,
    pool_status_root: Path | None = None,
) -> list[EnvServerSpec]:
    specs: list[EnvServerSpec] = []
    for local_index in range(servers_per_node):
        replica_index = replica_offset + local_index
        port = base_port + node_rank * servers_per_node + local_index
        name = f"{name_prefix}-{replica_index:04d}"
        pool_status_dir = (
            str(Path(pool_status_root) / name) if pool_status_root is not None else None
        )
        specs.append(
            EnvServerSpec(
                name=name,
                host=host,
                port=port,
                bind_address=f"tcp://{bind_host}:{port}",
                public_address=f"tcp://{host}:{port}",
                num_workers=workers_per_server,
                node_rank=node_rank,
                local_index=local_index,
                replica_index=replica_index,
                replica_count=replica_count,
                config_path=str(config_dir / f"{name}.toml"),
                log_path=str(log_dir / f"{name}.log"),
                pool_status_dir=pool_status_dir,
            )
        )
    return specs


def _slurm_node_address(
    env: Mapping[str, str],
    *,
    run_command: Callable[[list[str]], str] | None,
    hostname_func: Callable[[], str],
) -> str | None:
    node_name = env.get("SLURMD_NODENAME") or hostname_func()
    if not node_name:
        return None

    candidates = [node_name]
    if "." in node_name:
        candidates.append(node_name.split(".", 1)[0])

    for candidate in dict.fromkeys(candidates):
        try:
            output = (
                run_command(["scontrol", "show", "node", candidate])
                if run_command is not None
                else _run_scontrol_show_node(candidate)
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if node_addr := parse_node_addr(output):
            return node_addr
    return None


def _run_scontrol_show_node(node_name: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "node", node_name],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def parse_node_addr(output: str) -> str | None:
    """Extract ``NodeAddr=`` from ``scontrol show node`` output."""
    for node_field in output.split():
        if not node_field.startswith("NodeAddr="):
            continue
        node_addr = node_field.split("=", 1)[1]
        if node_addr and node_addr != "(null)":
            return node_addr
    return None


@dataclass(frozen=True)
class FleetRunLayout:
    run_id: str
    run_base: Path
    run_root: Path
    registry_path: Path
    pool_root: Path
    pool_status_dir: Path
    logs_dir: Path
    configs_dir: Path
    consumer_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.run_base / self.run_id

    @classmethod
    def for_run(
        cls,
        *,
        run_id: str,
        run_base: str | Path,
        run_root: str | Path | None = None,
        registry_path: str | Path | None = None,
        pool_root: str | Path | None = None,
        pool_status_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
        configs_dir: str | Path | None = None,
        consumer_paths: Mapping[str, str | Path | None] | None = None,
    ) -> Self:
        resolved_run_id = str(run_id)
        resolved_run_base = require_absolute_path(run_base, name="run_base")
        resolved_run_root = Path(
            run_root or resolved_run_base / resolved_run_id / "env_fleet"
        )
        resolved_pool_root = Path(
            pool_root or resolved_run_base / resolved_run_id / DEFAULT_DESKTOP_POOL_DIR
        )
        return cls(
            run_id=resolved_run_id,
            run_base=resolved_run_base,
            run_root=require_absolute_path(resolved_run_root, name="run_root"),
            registry_path=require_absolute_path(
                registry_path or resolved_run_root / "env_registry.json",
                name="registry_path",
            ),
            pool_root=require_absolute_path(resolved_pool_root, name="pool_root"),
            pool_status_dir=require_absolute_path(
                pool_status_dir or resolved_pool_root / "status",
                name="pool_status_dir",
            ),
            logs_dir=require_absolute_path(
                logs_dir or resolved_run_root / "logs",
                name="logs_dir",
            ),
            configs_dir=require_absolute_path(
                configs_dir or resolved_run_root / "configs",
                name="configs_dir",
            ),
            consumer_paths=_consumer_paths(consumer_paths),
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] = os.environ,
        *,
        run_id: str | None = None,
        run_base: str | Path | None = None,
    ) -> Self:
        resolved_run_id = run_id or env.get("OSWORLD_FLEET_RUN_ID") or slurm_run_id(env)
        if run_base is not None:
            resolved_run_base = require_absolute_path(run_base, name="run_base")
        elif env.get("OSWORLD_RUN_BASE"):
            resolved_run_base = require_absolute_path(
                env["OSWORLD_RUN_BASE"],
                name="OSWORLD_RUN_BASE",
            )
        else:
            resolved_run_base = scratch_root(env)
        return cls.for_run(
            run_id=resolved_run_id,
            run_base=resolved_run_base,
            run_root=env_path(env, "OSWORLD_FLEET_RUN_ROOT"),
            registry_path=env_path(env, "OSWORLD_ENV_FLEET_REGISTRY"),
            pool_root=env_path(env, "OSWORLD_DESKTOP_POOL_ROOT"),
            pool_status_dir=env_path(env, "OSWORLD_DESKTOP_POOL_STATUS_DIR"),
            logs_dir=env_path(env, "OSWORLD_FLEET_LOGS_DIR"),
            configs_dir=env_path(env, "OSWORLD_FLEET_CONFIGS_DIR"),
            consumer_paths=parse_consumer_paths(env.get(CONSUMER_PATHS_ENV)),
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        fallback: Self | None = None,
    ) -> Self | None:
        layout = metadata.get("layout")
        if not isinstance(layout, Mapping):
            return fallback

        def _value(name: str, default: object | None = None) -> object | None:
            return layout.get(name, default)

        run_id = _value("run_id", fallback.run_id if fallback else None)
        run_base = _value("run_base", fallback.run_base if fallback else None)
        if run_id is None or run_base is None:
            return fallback
        consumer_paths = {
            str(key): Path(str(value))
            for key, value in layout.items()
            if key not in LAYOUT_KEYS and value
        }
        if fallback is not None:
            consumer_paths = {**fallback.consumer_paths, **consumer_paths}
        return cls.for_run(
            run_id=str(run_id),
            run_base=Path(str(run_base)),
            run_root=_path_value(_value("run_root")),
            registry_path=_path_value(_value("registry_path")),
            pool_root=_path_value(_value("pool_root")),
            pool_status_dir=_path_value(_value("pool_status_dir")),
            logs_dir=_path_value(_value("logs_dir")),
            configs_dir=_path_value(_value("configs_dir")),
            consumer_paths=consumer_paths,
        )

    def as_metadata(self) -> dict[str, str]:
        metadata = {key: str(getattr(self, key)) for key in LAYOUT_KEYS}
        metadata.update(
            {key: str(value) for key, value in sorted(self.consumer_paths.items())}
        )
        return metadata

    def consumer_path(self, name: str, *, default: str | Path | None = None) -> Path:
        """Read one opaque consumer path, falling back to a caller-supplied default."""
        value = self.consumer_paths.get(name)
        if value is not None:
            return value
        if default is None:
            raise KeyError(f"layout has no consumer path {name!r}")
        return require_absolute_path(default, name=name)

    def with_consumer_paths(
        self,
        consumer_paths: Mapping[str, str | Path | None],
    ) -> Self:
        return replace(
            self,
            consumer_paths={
                **self.consumer_paths,
                **_consumer_paths(consumer_paths),
            },
        )

    def node_configs_dir(self, node_rank: int) -> Path:
        return self.configs_dir / f"node_{node_rank:04d}"

    def node_logs_dir(self, node_rank: int) -> Path:
        return self.logs_dir / f"node_{node_rank:04d}"


# Layout keys desktop-fleet itself owns, in field order. Anything else found under
# the registry's ``layout`` metadata is a consumer path and is carried verbatim.
# Derived from the dataclass so a new layout field cannot silently start being
# read back as somebody's consumer path.
LAYOUT_KEYS: tuple[str, ...] = tuple(
    f.name for f in fields(FleetRunLayout) if f.name != "consumer_paths"
)


def parse_consumer_paths(value: str | None) -> dict[str, Path]:
    """Parse ``name=/abs/path,name=/abs/path`` into absolute consumer paths."""
    if not value:
        return {}
    parsed: dict[str, Path] = {}
    for item in value.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{CONSUMER_PATHS_ENV} entries must be name=path: {entry!r}")
        name, raw_path = entry.split("=", 1)
        parsed[name.strip()] = require_absolute_path(raw_path.strip(), name=name.strip())
    return parsed


def render_consumer_paths(consumer_paths: Mapping[str, str | Path]) -> str:
    """Render consumer paths back into the ``ENV_FLEET_CONSUMER_PATHS`` form."""
    return ",".join(f"{key}={value}" for key, value in sorted(consumer_paths.items()))


def _consumer_paths(
    consumer_paths: Mapping[str, str | Path | None] | None,
) -> dict[str, Path]:
    if not consumer_paths:
        return {}
    return {
        str(key): require_absolute_path(value, name=str(key))
        for key, value in consumer_paths.items()
        if value is not None
    }


def _path_value(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def env_path(env: Mapping[str, str], name: str) -> Path | None:
    """Read one absolute path override out of the environment."""
    value = env.get(name)
    if value is None:
        return None
    return require_absolute_path(value, name=name)


def write_env_server_config(
    spec: EnvServerSpec,
    args: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Render the ``verifiers`` env-server TOML for one replica."""
    harness = deepcopy(metadata["harness"])
    if spec.pool_status_dir:
        desktop = harness.setdefault("desktop", {})
        pool_config = desktop.setdefault("desktop_pool_config", {})
        pool_config["status_dir"] = spec.pool_status_dir
    taskset = {
        "id": args.env_id,
        "base_path": str(args.task_base_path),
        "max_tasks": args.max_tasks,
        "shuffle_seed": args.shuffle_seed,
    }
    harness["id"] = args.env_id
    payload = {
        "output_dir": str(args.run_root / "server_output"),
        "log": {"level": "INFO"},
        "env": {
            "name": spec.name,
            "address": spec.bind_address,
            "taskset": taskset,
            "harness": harness,
            "pool": {"type": "static", "num_workers": spec.num_workers},
            "timeout": {"rollout": args.rollout_timeout},
            "retries": {"rollout": {"max_retries": args.env_max_retries}},
            "max_turns": args.max_steps,
        },
    }
    Path(spec.config_path).write_text(to_toml(payload), encoding="utf-8")


def to_toml(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    scalar_items = {
        key: value for key, value in payload.items() if not isinstance(value, Mapping)
    }
    for key, value in scalar_items.items():
        lines.append(f"{key} = {toml_literal(value)}")
    for section, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, item in value.items():
            lines.append(f"{key} = {toml_literal(item)}")
    return "\n".join(lines).strip() + "\n"


def toml_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{key} = {toml_literal(item)}" for key, item in value.items()
        )
        return f"{{ {items} }}"
    if isinstance(value, list):
        return "[" + ", ".join(toml_literal(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")
