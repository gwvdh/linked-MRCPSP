from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Optional

from .definitions import Process
from .generator import active_phases, get_capacity, make_phase_timelines

PhaseTimeline = dict[int, dict[int, int]]


def greedy_schedule(
    processes: list[Process],
    max_phases: int,
    capacities: dict[int, int],
) -> tuple[list[PhaseTimeline], int]:
    """
    Greedy earliest-start schedule subject to renewable resource capacities.
    """
    phase_timelines = make_phase_timelines(max_phases)
    usage: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    horizon_ub = sum(p.max_processing_time() for p in processes) + 1
    makespan = 0

    for process in processes:
        modes = [
            random.randrange(process.phases[i].number_of_modes)
            for i in range(len(process.phases))
        ]

        phase_end: list[int] = []

        for phase_idx in range(active_phases(process.network_type)):
            predecessors = process.network_type.value[phase_idx]
            t = process.start_time if not predecessors else max(phase_end[p] for p in predecessors)

            for task in process.tasks[phase_idx]:
                duration = task.duration[modes[phase_idx]]
                resource = task.resource[modes[phase_idx]]

                if resource is not None:
                    cap = capacities.get(resource, float("inf"))
                    while any(usage[t + dt][resource] >= cap for dt in range(duration)):
                        t += 1
                        if t > horizon_ub:
                            break

                    for dt in range(duration):
                        usage[t + dt][resource] += 1
                        phase_timelines[phase_idx][t + dt][resource] += 1

                t += duration

            phase_end.append(t)

        if phase_end:
            makespan = max(makespan, max(phase_end))

    return [dict(sorted(tl.items())) for tl in phase_timelines], makespan


def estimate_time_horizon(
    processes: list[Process],
    max_phases: int,
    capacities: dict[int, int],
    trials: int = 20,
) -> int:
    """
    Estimate a feasible horizon by repeated greedy schedules.
    """
    best = sum(p.max_processing_time() for p in processes) + 1
    for _ in range(trials):
        _, makespan = greedy_schedule(processes, max_phases, capacities)
        best = min(best, makespan + 1)
    print(f"Estimated T = {best}")
    return best


def get_or_instance(
    processes: list[Process],
    scarcity: float,
    max_start_time: int,
    n_resources: int,
    min_max: list[tuple[int, int]],
    max_phases: int = 3,
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

    max_start_time = estimate_time_horizon(
        processes=processes,
        max_phases=max_phases,
        capacities={r: capacities[r] for r in range(n_resources)},
        trials=100,
    )

    def zero_p(n_modes: int) -> list[int]:
        return [0] * n_modes

    def zero_r(n_modes: int) -> list[list[int]]:
        return [[0] * n_resources for _ in range(n_modes)]

    source = 0
    next_job = 1

    p: list[list[int]] = [zero_p(1)]
    r: list[list[list[int]]] = [zero_r(1)]
    M: list[int] = [1]
    ES: list[int] = [0]
    E: list[list[int]] = []
    L: list[list[int]] = []
    O: list[int] = []

    for process in processes:
        n_active = active_phases(process.network_type)
        phase_last: list[Optional[int]] = [None] * n_active

        for phase_idx in range(n_active):
            phase_tasks = process.tasks[phase_idx]
            n_modes = process.phases[phase_idx].number_of_modes

            for task_idx, task in enumerate(phase_tasks):
                current = next_job

                if task_idx == 0:
                    predecessors = process.network_type.value[phase_idx]
                    if not predecessors:
                        E.append([source, current])
                    else:
                        for pred_phase in predecessors:
                            pred_last = phase_last[pred_phase]
                            if pred_last is not None:
                                E.append([pred_last, current])
                else:
                    E.append([current - 1, current])

                if task_idx > 0:
                    L.append([current - 1, current])

                p.append(zero_p(n_modes))
                r.append(zero_r(n_modes))
                M.append(n_modes)
                ES.append(process.start_time)

                for m in range(n_modes):
                    p[current][m] = task.duration[m]
                    res = task.resource[m]
                    if res is not None:
                        r[current][m][res] = 1

                if task_idx == len(phase_tasks) - 1:
                    phase_last[phase_idx] = current

                next_job += 1

        final_phase = n_active - 1
        if phase_last[final_phase] is not None:
            O.append(phase_last[final_phase])

    sink = next_job
    p.append(zero_p(1))
    r.append(zero_r(1))
    M.append(1)
    ES.append(0)

    for last_job in O:
        E.append([last_job, sink])

    n = sink + 1

    # Transitive closure
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in E:
        adj[i].add(j)

    for k in range(n):
        for i in range(n):
            if k in adj[i]:
                adj[i] |= adj[k]

    TE = [[i, j] for i in range(n) for j in adj[i]]

    # Earliest start bounds
    for i in range(n):
        for k in range(n):
            if i in adj[k]:
                ES[i] = max(ES[i], ES[k] + min(p[k]))

    # Latest start bounds
    LS = [int(max_start_time) - 1] * n
    for i in range(n - 1, -1, -1):
        for k in range(n):
            if k in adj[i]:
                LS[i] = min(LS[i], LS[k] - min(p[i]))

    comparable = {(i, j) for i, j in TE} | {(j, i) for i, j in TE}
    VP = [[i, j] for i in range(n) for j in range(n) if i != j and (i, j) not in comparable]

    return {
        "n": n,
        "T": int(max_start_time),
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
