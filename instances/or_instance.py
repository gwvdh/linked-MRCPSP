from __future__ import annotations

from typing import Any

from .definitions import Process
from .generator import get_capacity


def get_or_instance(
    processes: list[Process],
    scarcity: float,
    n_resources: int,
    min_max: list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Build an OR instance with:
      n   number of jobs including source and sink
      T   time horizon
      M   number of modes per job
      R   resource capacities
      E   precedence arcs
      L   linked-mode arcs
      p   processing times
      r   resource requirements
      O   last real job per process
      ES  earliest start times
      LS  latest start times
      VP  incomparable job pairs
    """
    if len(min_max) != n_resources:
        raise ValueError(
            f"Expected {n_resources} (min, max) pairs, got {len(min_max)}"
        )

    capacities = [get_capacity(mn, mx, scarcity) for mn, mx in min_max]
    print(f"Capacities: {capacities}")

    # Serially schedule all jobs after the latest release to obtain a safe horizon.
    horizon = (
        max((process.start_time for process in processes), default=0)
        + sum(process.max_processing_time() for process in processes)
        + 1
    )

    source = 0
    next_job = 1

    p: list[list[int]] = [[0]]
    r: list[list[list[int]]] = [[[0] * n_resources]]
    M: list[int] = [1]
    ES: list[int] = [0]
    E: list[list[int]] = []
    L: list[list[int]] = []
    O: list[int] = []

    # Connect each phase's first task to its predecessors. Link tasks within a phase to share a mode.
    for process in processes:
        n_active = process.active_phases()
        phase_last: list[int | None] = [None] * n_active

        for phase_idx in range(n_active):
            phase_tasks = process.tasks[phase_idx]
            n_modes = process.phases[phase_idx].number_of_modes
            predecessors = process.network_type.value[phase_idx]
            previous_task = None

            for task in phase_tasks:
                current = next_job

                if previous_task is not None:
                    E.append([previous_task, current])
                    L.append([previous_task, current])
                elif not predecessors:
                    E.append([source, current])
                else:
                    E.extend(
                        [phase_last[pred_phase], current]
                        for pred_phase in predecessors
                        if phase_last[pred_phase] is not None
                    )

                p.append(list(task.duration))
                r.append(
                    [
                        [int(resource == resource_idx) for resource_idx in range(n_resources)]
                        for resource in task.resource
                    ]
                )
                M.append(n_modes)
                ES.append(process.start_time)

                previous_task = current
                next_job += 1

            phase_last[phase_idx] = previous_task

        final_phase = n_active - 1
        if phase_last[final_phase] is not None:
            O.append(phase_last[final_phase])

    sink = next_job
    p.append([0])
    r.append([[0] * n_resources])
    M.append(1)
    ES.append(0)

    for last_job in O:
        E.append([last_job, sink])

    n = sink + 1

    # Compute reachability for time bounds and incomparable task pairs.
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in E:
        adj[i].add(j)

    for k in range(n):
        for i in range(n):
            if k in adj[i]:
                adj[i] |= adj[k]

    TE = [[i, j] for i in range(n) for j in adj[i]]

    # Use shortest mode durations to derive bounds valid for every mode choice.
    for i in range(n):
        for k in range(n):
            if i in adj[k]:
                ES[i] = max(ES[i], ES[k] + min(p[k]))

    # Latest start bounds
    LS = [horizon - 1] * n
    for i in range(n - 1, -1, -1):
        for k in range(n):
            if k in adj[i]:
                LS[i] = min(LS[i], LS[k] - min(p[i]))

    comparable = {(i, j) for i, j in TE} | {(j, i) for i, j in TE}
    VP = [[i, j] for i in range(n) for j in range(n) if i != j and (i, j) not in comparable]

    return {
        "n": n,
        "T": horizon,
        "M": M,
        "R": capacities,
        "E": E,
        "L": L,
        "p": p,
        "r": r,
        "O": O,
        "ES": ES,
        "LS": LS,
        "VP": VP,
    }
