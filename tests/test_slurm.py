from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from env_fleet import slurm
from env_fleet.slurm import (
    SlurmJob,
    confirm_cancel,
    parse_squeue,
    query_squeue,
    run_command,
    select_cancel_job,
    slurm_node_addrs,
)


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_parse_squeue_and_cancel_guards():
    job = SlurmJob(
        job_id="12345",
        user="user-a",
        name="osworld_env_fleet",
        state="RUNNING",
        reason="None",
        elapsed="1:00",
        nodes="1",
        cpus="16",
    )

    assert parse_squeue("12345|user-a|osworld_env_fleet|RUNNING|None|1:00|1|16|\n") == [
        job
    ]
    assert select_cancel_job([job], user="user-a", job_name="osworld_env_fleet") == job
    assert confirm_cancel(job, yes=False, input_fn=lambda _prompt: "n") is False
    assert confirm_cancel(job, yes=True) is True


def test_select_cancel_job_refuses_foreign_or_ambiguous_jobs():
    job = SlurmJob(
        job_id="12345",
        user="user-a",
        name="osworld_env_fleet",
        state="RUNNING",
        reason="None",
    )

    assert select_cancel_job([], user="user-a", job_name="osworld_env_fleet") is None
    with pytest.raises(ValueError, match="multiple jobs matched"):
        select_cancel_job([job, job], user="user-a", job_name="osworld_env_fleet")
    with pytest.raises(ValueError, match="owner is user-a"):
        select_cancel_job([job], user="user-b", job_name="osworld_env_fleet")
    with pytest.raises(ValueError, match="name is osworld_env_fleet"):
        select_cancel_job([job], user="user-a", job_name="other_fleet")


def _fake_completed_process(*, returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_command_returns_stdout_and_exact_argv_on_success(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _fake_completed_process(returncode=0, stdout="NodeAddr=10.0.0.1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_command(["scontrol", "show", "node", "node001"])

    assert result == "NodeAddr=10.0.0.1\n"
    assert calls == [["scontrol", "show", "node", "node001"]]


def test_run_command_does_not_leak_stdout_from_a_nonzero_exit(monkeypatch):
    """A failing scontrol call must not surface as if it had succeeded."""

    def fake_run(command, **kwargs):
        return _fake_completed_process(
            returncode=1,
            stdout="NodeAddr=looks-plausible-but-is-not-real\n",
            stderr="scontrol: error: invalid node name",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_command(["scontrol", "show", "node", "bogus"]) == ""


def test_run_command_swallows_a_missing_binary_without_raising(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("scontrol: command not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_command(["scontrol", "show", "node", "node001"]) == ""


def test_query_squeue_uses_jobs_filter_when_job_id_is_given(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _fake_completed_process(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    query_squeue(job_id="12345", job_name="osworld_env_fleet")

    assert calls == [
        [
            "squeue",
            "--noheader",
            "--me",
            "--format=%i|%u|%j|%T|%R|%M|%D|%C|%b",
            "--jobs",
            "12345",
        ]
    ]


def test_query_squeue_uses_name_filter_when_job_id_is_absent(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _fake_completed_process(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    query_squeue(job_id=None, job_name="osworld_env_fleet")

    assert calls == [
        [
            "squeue",
            "--noheader",
            "--me",
            "--format=%i|%u|%j|%T|%R|%M|%D|%C|%b",
            "--name",
            "osworld_env_fleet",
        ]
    ]


def test_query_squeue_returns_empty_list_on_nonzero_exit_not_a_fabricated_job(
    monkeypatch,
):
    """A failing squeue call must not parse whatever garbage landed on stdout."""

    def fake_run(command, **kwargs):
        return _fake_completed_process(
            returncode=1,
            stdout="12345|user-a|osworld_env_fleet|RUNNING|None|1:00|1|16|\n",
            stderr="squeue: error: Invalid user",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert query_squeue(job_id=None, job_name="osworld_env_fleet") == []


def test_slurm_node_addrs_returns_empty_without_calling_scontrol(monkeypatch):
    calls = []
    monkeypatch.setattr(slurm, "run_command", lambda command: calls.append(command) or "")

    assert slurm_node_addrs({}) == []
    assert calls == []


def test_slurm_node_addrs_expands_a_node_range_and_resolves_each_hostname(monkeypatch):
    calls = []

    def fake_run_command(command):
        calls.append(command)
        if command == ["scontrol", "show", "hostnames", "node[001-003]"]:
            return "node001\nnode002\nnode003\n"
        if command == ["scontrol", "show", "node", "node001"]:
            return "NodeName=node001 NodeAddr=10.0.0.1 NodeHostName=node001\n"
        if command == ["scontrol", "show", "node", "node002"]:
            return "NodeName=node002 NodeAddr=10.0.0.2 NodeHostName=node002\n"
        if command == ["scontrol", "show", "node", "node003"]:
            # simulate a failed/empty lookup for this one node only
            return ""
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    addresses = slurm_node_addrs({"SLURM_JOB_NODELIST": "node[001-003]"})

    # the raw range expression is passed through verbatim -- expansion is scontrol's job
    assert calls[0] == ["scontrol", "show", "hostnames", "node[001-003]"]
    assert calls[1:] == [
        ["scontrol", "show", "node", "node001"],
        ["scontrol", "show", "node", "node002"],
        ["scontrol", "show", "node", "node003"],
    ]
    # node003's failed lookup falls back to the raw hostname rather than being dropped
    assert addresses == ["10.0.0.1", "10.0.0.2", "node003"]


def test_slurm_node_addrs_ignores_blank_hostname_lines(monkeypatch):
    def fake_run_command(command):
        if command == ["scontrol", "show", "hostnames", "node[001-002]"]:
            return "node001\n\n   \nnode002\n"
        if command == ["scontrol", "show", "node", "node001"]:
            return "NodeAddr=10.0.0.1\n"
        if command == ["scontrol", "show", "node", "node002"]:
            return "NodeAddr=10.0.0.2\n"
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    addresses = slurm_node_addrs({"SLURM_JOB_NODELIST": "node[001-002]"})

    # blank/whitespace-only lines must not become spurious entries in the result
    assert addresses == ["10.0.0.1", "10.0.0.2"]
