from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import takewhile
from typing import Optional

import numpy as np

from .xml_parser import RA_PST


class NetworkType(Enum):
    """Adjacency list of direct predecessors for each phase."""

    SINGLE = ((),)
    DOUBLE = ((), (0,))
    TRIPLE = ((), (0,), (1,))
    INTREE = ((), (), (0, 1))


@dataclass
class Task:
    duration: list[int]
    resource: list[int | None]
    start_time: int


class PhaseProfile:
    """
    Each phase consists of a chain of tasks, where each task has a duration and a resource requirement.
    The resource requirement is None if the task is not assigned to a resource.
    """
    def __init__(
        self,
        base_durations: list[float],
        resource_ratios: dict[str, float],
        ra_pst: RA_PST,
    ):
        self.base_durations = base_durations
        self.resource_ratios = resource_ratios
        self.number_of_tasks = ra_pst.get_number_of_tasks()
        self.resource_levels = ra_pst.get_number_of_resources()
        self.number_of_modes = ra_pst.get_number_of_modes()
        self.ra_pst = ra_pst

        assert len(base_durations) == self.number_of_tasks, (
            f"Expected {self.number_of_tasks} base durations, "
            f"got {len(base_durations)}"
        )

    def get_task(self, mode: int, task_id: int):
        return self.ra_pst.paths[mode][task_id]

    def get_duration(self, task_id: int, mode: int) -> float:
        assert 0 <= task_id < self.number_of_tasks
        assert 0 <= mode < self.number_of_modes
        node = self.ra_pst.paths[mode][task_id]
        multiplier = self.resource_ratios.get(node.resource_id, 1.0) # default ratio 1.0
        return self.base_durations[task_id] * multiplier

    def __str__(self) -> str:
        return f"PhaseProfile(ratios={self.resource_ratios})"


class Process:
    def __init__(
        self,
        network_type: NetworkType,
        phases: list[PhaseProfile],
        start_time: int = 0,
    ):
        self.network_type = network_type
        self.phases = phases
        self.start_time = start_time
        self.tasks = self._define_tasks(variance=0.0)

    def max_processing_time(self) -> int:
        """Return the max processing time of the process in any mode."""
        duration = 0
        for i, phase in enumerate(self.phases):
            phase_duration = 0
            for mode in range(phase.number_of_modes):
                phase_duration = max(phase_duration, sum(
                                    task.duration[mode]
                                    for task in self.tasks[i]
                                    if task is not None
                                    ))
            duration += phase_duration
        return duration

    def get_max_resource_demand_mode(self, resource: int) -> list[int | None]:
        n_phases = len(self.phases)
        best_mode: list[int | None] = [None] * n_phases
        best_demand = [float("-inf")] * n_phases

        for phase_id, phase_tasks in enumerate(self.tasks):
            for mode in range(self.phases[phase_id].number_of_modes):
                demand = sum(
                    task.duration[mode]
                    for task in phase_tasks
                    if task is not None and task.resource[mode] == resource
                )
                if demand > best_demand[phase_id]:
                    best_demand[phase_id] = demand
                    best_mode[phase_id] = mode

        return best_mode

    def _define_tasks(self, variance: float = 0.0) -> list[list[Task]]:
        """
        Generate a list of tasks for each phase with randomized durations.
        If subsequent tasks are dummy tasks, add the base durations to the current task.
        Any dummy task has resource requirement None and duration 0.
        :param variance: The standard deviation of the task durations for added noise
        """
        tasks: list[list[Task]] = []

        for phase in self.phases:
            phase_tasks: list[Task] = []

            for j in range(phase.number_of_tasks):
                durations: list[int] = []
                resources: list[int | None] = []

                for m in range(phase.number_of_modes):
                    if phase.get_task(m, j).is_dummy:
                        durations.append(0)
                        resources.append(None)
                        continue

                    duration = phase.get_duration(j, m) + sum(
                        phase.get_duration(next_j, m)
                        for next_j in takewhile(  # add durations of subsequent dummy tasks
                            lambda k: phase.get_task(m, k).is_dummy,
                            range(j + 1, phase.number_of_tasks),
                        )
                    )

                    if variance > 0.0:
                        duration = max(1, duration*np.random.normal(1.0, variance))

                    durations.append(int(duration))
                    resources.append(phase.ra_pst.get_resource(j, m))

                phase_tasks.append(
                    Task(durations, resources, start_time=self.start_time)
                )

            tasks.append(phase_tasks)

        return tasks

    def __str__(self) -> str:
        return (
            f"Process(network={self.network_type}, "
            f"start={self.start_time}, phases={len(self.phases)})"
        )
