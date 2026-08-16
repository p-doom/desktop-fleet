"""Consumer-specific coupling lives here and nowhere else.

Every module under ``desktop_fleet.adapters`` may name a particular trainer,
orchestrator, or RL framework. Every module *outside* it must not: the core
speaks only of machines, registries, capacity, and Slurm. Import direction is
strictly adapter -> core; the core never imports an adapter.
"""

from __future__ import annotations

__all__: list[str] = []
