"""Slurm as the fleet's scheduler: identity, node addressing, queue queries.

The scarce resource an desktop-fleet manages is an externally-leased machine, and on
this cluster machines are leased from Slurm. Everything Slurm-shaped lives here.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from desktop_fleet.registry import EnvFleetRegistry
from desktop_fleet.spec import parse_node_addr


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    user: str
    name: str
    state: str
    reason: str
    elapsed: str = ""
    nodes: str = ""
    cpus: str = ""
    tres: str = ""


def slurm_metadata(env: Mapping[str, str]) -> dict[str, str]:
    """Capture static Slurm identity for operator tooling."""
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NNODES",
        "SLURM_CPUS_PER_TASK",
    ]
    return {key.lower(): value for key in keys if (value := env.get(key))}


def slurm_node_addrs(env: Mapping[str, str]) -> list[str]:
    """Resolve Slurm node names to NodeAddr values when available."""
    nodelist = env.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return []
    hostnames = run_command(["scontrol", "show", "hostnames", nodelist]).splitlines()
    addresses: list[str] = []
    for hostname in hostnames:
        node_name = hostname.strip()
        if not node_name:
            continue
        node_info = run_command(["scontrol", "show", "node", node_name])
        addresses.append(parse_node_addr(node_info) or node_name)
    return addresses


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def query_squeue(*, job_id: str | None, job_name: str) -> list[SlurmJob]:
    command = [
        "squeue",
        "--noheader",
        "--me",
        "--format=%i|%u|%j|%T|%R|%M|%D|%C|%b",
    ]
    if job_id:
        command.extend(["--jobs", str(job_id)])
    elif job_name:
        command.extend(["--name", job_name])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return parse_squeue(result.stdout)


def parse_squeue(output: str) -> list[SlurmJob]:
    jobs: list[SlurmJob] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        padded = [*fields, *[""] * (9 - len(fields))]
        jobs.append(
            SlurmJob(
                job_id=padded[0],
                user=padded[1],
                name=padded[2],
                state=padded[3],
                reason=padded[4],
                elapsed=padded[5],
                nodes=padded[6],
                cpus=padded[7],
                tres=padded[8],
            )
        )
    return jobs


def slurm_job_id_from_registry(registry: EnvFleetRegistry) -> str | None:
    """Read the job id ``slurm_metadata`` recorded, or nothing.

    Never guessed from ``run_id`` looking like a number: an id that was not
    recorded by the node that holds the allocation is not a job id, and
    ``scancel`` on a guess cancels a stranger's job.
    """
    slurm = registry.metadata.get("slurm")
    if not isinstance(slurm, Mapping):
        return None
    value = slurm.get("slurm_job_id")
    return str(value) if value else None


def select_cancel_job(
    jobs: Sequence[SlurmJob],
    *,
    user: str,
    job_name: str,
) -> SlurmJob | None:
    if not jobs:
        return None
    if len(jobs) > 1:
        raise ValueError("Refusing to cancel because multiple jobs matched.")
    job = jobs[0]
    if job.user != user:
        raise ValueError(f"Refusing to cancel job {job.job_id}; owner is {job.user}.")
    if job.name != job_name:
        raise ValueError(f"Refusing to cancel job {job.job_id}; name is {job.name}.")
    return job


def confirm_cancel(
    job: SlurmJob,
    *,
    yes: bool,
    input_fn: Callable[[str], str] = input,
) -> bool:
    if yes:
        return True
    answer = input_fn(
        f"Cancel fleet job {job.job_id} ({job.state}, {job.reason})? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}
