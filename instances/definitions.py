from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import takewhile
from typing import Optional

import numpy as np

from .xml_parser import RA_PST


class NetworkType(Enum):
    """
    Precedence between phases, encoded as direct predecessors per phase.
    """

    SINGLE = ((),)
    DOUBLE = ((), (0,))
    TRIPLE = ((), (0,), (1,))
    INTREE = ((), (), (0, 1))


@dataclass
class Task:
    duration: list[int]
    resource: list[int | None]
    start_time: int = 0


class PhaseProfile:
    """
    A phase is a task chain. Each task may have multiple modes inherited from
    the RA-PST. Durations are scaled by resource-specific ratios.
    """

    def __init__(
        self,
        base_durations: list[float],
        resource_ratios: dict[str, float],
        ra_pst: RA_PST,
    ):
        self.base_durations = base_durations
        self.resource_ratios = resource_ratios
        self.ra_pst = ra_pst

        self.number_of_tasks = ra_pst.get_number_of_tasks()
        self.number_of_modes = ra_pst.get_number_of_modes()
        self.number_of_resources = ra_pst.get_number_of_resources()

        if len(base_durations) != self.number_of_tasks:
            raise ValueError(
                f"Expected {self.number_of_tasks} base durations, "
                f"got {len(base_durations)}"
            )

    def get_task(self, mode: int, task_id: int):
        return self.ra_pst.paths[mode][task_id]

    def get_duration(self, task_id: int, mode: int) -> float:
        node = self.get_task(mode, task_id)
        factor = self.resource_ratios.get(node.resource_id, 1.0)
        return self.base_durations[task_id] * factor

    def __str__(self) -> str:
        return f"PhaseProfile(modes={self.number_of_modes})"


class Process:
    """
    A process consists of several phases connected by a small phase network.
    """

    def __init__(
        self,
        network_type: NetworkType,
        phases: list[PhaseProfile],
        start_time: int = 0,
        variance: float = 0.0,
    ):
        self.network_type = network_type
        self.phases = phases
        self.start_time = start_time
        self.tasks = self._build_tasks(variance=variance)

    def active_phases(self) -> int:
        return len(self.network_type.value)

    def max_processing_time(self) -> int:
        total = 0
        for phase_idx, phase in enumerate(self.phases):
            total += max(
                sum(task.duration[m] for task in self.tasks[phase_idx])
                for m in range(phase.number_of_modes)
            )
        return total

    def get_max_resource_demand_mode(self, resource: int) -> list[int | None]:
        best_mode: list[int | None] = [None] * len(self.phases)
        best_demand = [float("-inf")] * len(self.phases)

        for phase_idx, phase_tasks in enumerate(self.tasks):
            n_modes = self.phases[phase_idx].number_of_modes
            for mode in range(n_modes):
                demand = sum(
                    task.duration[mode]
                    for task in phase_tasks
                    if task.resource[mode] == resource
                )
                if demand > best_demand[phase_idx]:
                    best_demand[phase_idx] = demand
                    best_mode[phase_idx] = mode

        return best_mode

    def _build_tasks(self, variance: float = 0.0) -> list[list[Task]]:
        """
        Build per-phase task chains.

        Dummy tasks have duration 0 and no resource. If a task is followed by
        consecutive dummy tasks in the same mode, their durations are merged
        into the current real task.
        """
        rng = np.random.default_rng()
        all_tasks: list[list[Task]] = []

        for phase in self.phases:
            phase_tasks: list[Task] = []

            for j in range(phase.number_of_tasks):
                durations: list[int] = []
                resources: list[int | None] = []

                for m in range(phase.number_of_modes):
                    node = phase.get_task(m, j)

                    if node.is_dummy:
                        durations.append(0)
                        resources.append(None)
                        continue

                    duration = phase.get_duration(j, m)
                    duration += sum(
                        phase.get_duration(k, m)
                        for k in takewhile(
                            lambda x: phase.get_task(m, x).is_dummy,
                            range(j + 1, phase.number_of_tasks),
                        )
                    )

                    if variance > 0.0:
                        duration = max(1.0, duration * rng.normal(1.0, variance))

                    durations.append(int(duration))
                    resources.append(phase.ra_pst.get_resource(j, m))

                phase_tasks.append(
                    Task(
                        duration=durations,
                        resource=resources,
                        start_time=self.start_time,
                    )
                )

            all_tasks.append(phase_tasks)

        return all_tasks

    def __str__(self) -> str:
        return (
            f"Process(network={self.network_type.name}, "
            f"start={self.start_time}, phases={len(self.phases)})"
        )
