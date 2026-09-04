"""The node-local runtime root, and the marker proving this task owns it.

QEMU workdirs and AF_UNIX sockets belong on node-local disk, not the shared
filesystem, so each task derives one root under its own job and node rank. That
root is wiped at startup and removed at shutdown, which is only safe if the
task can prove it created it -- hence the owner marker, and hence a path that
is derived here rather than accepted from a caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOCAL_RUNTIME_TMP_ROOT_ENV = "ENV_FLEET_DESKTOP_POOL_TMP_ROOT"
LOCAL_RUNTIME_OWNER_FILE = ".desktop-fleet-runtime-owner.json"


@dataclass(frozen=True, slots=True)
class LocalRuntimeScope:
    base: Path
    job_id: str
    node_rank: int

    @property
    def job_root(self) -> Path:
        return self.base / f"desktop-fleet-{self.job_id}"

    @property
    def node_root(self) -> Path:
        return self.job_root / f"node-{self.node_rank}"

    @property
    def runtime_root(self) -> Path:
        return self.node_root / "desktop-runtime"

    @property
    def owner_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "job_id": self.job_id,
            "node_rank": self.node_rank,
            "runtime_root": str(self.runtime_root),
        }


def local_runtime_scope(env: Mapping[str, str]) -> LocalRuntimeScope:
    """Derive one job- and node-owned runtime root from a configured base.

    ``SLURM_JOB_ID`` is required with no ``"manual"`` fallback, unlike
    :func:`desktop_fleet.spec.slurm_run_id`: two off-scheduler tasks sharing one
    id would share a root, and this one is removed, not just read.
    """
    base_value = env.get(LOCAL_RUNTIME_TMP_ROOT_ENV) or env.get("TMPDIR")
    job_id = env.get("SLURM_JOB_ID")
    if not base_value:
        raise RuntimeError(
            f"{LOCAL_RUNTIME_TMP_ROOT_ENV} or TMPDIR must be set "
            "for local runtime cleanup"
        )
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID must be set for local runtime cleanup")

    base = Path(base_value)
    if not base.is_absolute():
        raise RuntimeError(f"local runtime base must be absolute: {base}")
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise RuntimeError(f"Invalid SLURM_JOB_ID for runtime cleanup: {job_id!r}")

    node_rank_value = env.get("SLURM_PROCID", "0")
    try:
        node_rank = int(node_rank_value)
    except ValueError as error:
        raise RuntimeError(
            f"Invalid SLURM_PROCID for runtime cleanup: {node_rank_value!r}"
        ) from error
    if node_rank < 0:
        raise RuntimeError(
            f"Invalid SLURM_PROCID for runtime cleanup: {node_rank_value!r}"
        )

    return LocalRuntimeScope(base=base.resolve(), job_id=job_id, node_rank=node_rank)
