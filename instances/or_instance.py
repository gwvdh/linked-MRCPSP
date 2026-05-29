from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import random
import numpy as np

from .generator import get_capacity, _make_phase_timelines, _active_phases
from .definitions import Process, NetworkType
from .xml_parser import RA_PST


def greedy_schedule(
    processes: List[Process],
    max_phases: int,
    capacities: List[int, int],
) -> tuple[List[PhaseTimeline], int]:
    """ Schedule processes greedily, starting with the earliest start time. """
    phase_timelines = _make_phase_timelines(max_phases)
    resource_usage: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    upper_bound = sum(process.max_processing_time() for process in processes)
    makespan = 0
    for process in processes:
        modes = [
            random.randrange(process.phases[i].number_of_modes)
            for i in range(len(process.phases))
        ]
        phase_end: List[int] = []
        for i in range(_active_phases(process.network_type)):
            predecessors = process.network_type.value[i]
            t = process.start_time if not predecessors else max(phase_end[p] for p in predecessors)
            for task in process.tasks[i]:
                if task is None: continue
                dur, res = task.duration[modes[i]], task.resource[modes[i]]
                if res is not None:
                    cap = capacities.get(res, float("inf"))
                    while any(resource_usage[t + dt][res] >= cap for dt in range(dur)):
                        t += 1
                        if t > upper_bound: break
                    for dt in range(dur):
                        resource_usage[t + dt][res] += 1
                        phase_timelines[i][t + dt][res] += 1
                t += dur
            phase_end.append(t)
        if phase_end: 
            makespan = max(makespan, max(phase_end))
    return [dict(sorted(tl.items())) for tl in phase_timelines], makespan


def get_max_T(processes: List[Process], max_phases: int, capacities: Dict[int, int], k: int = 10) -> int:
    """ Return the minimum makespan over k greedy schedules. """
    max_T = float("inf")
    for _ in range(k):
        _, makespan = greedy_schedule(processes, max_phases, capacities)
        max_T = min(max_T, makespan+1)
    print(f"Max T = {max_T}")
    return max_T


def get_or_instance(
    processes: List[Process],
    scarcity: float,
    max_start_time: int,
    n_resources: int,
    min_max: List[Tuple[int, int]],  # per resource, shared across phases
    max_phases: int = 3,
) -> Dict[str, Any]:
    """
    Translate generated processes into an OR instance.

    Resources are shared across all phases: R[r] is the single capacity for
    resource r regardless of which phase a task belongs to.

    Returns a dict with keys: n, T, M, R, E, L, p, r, O, ES, VP.
    """
    assert len(min_max) == n_resources, (
        f"Expected {n_resources} (min, max) pairs, got {len(min_max)}"
    )

    # Single capacity vector — one entry per resource, shared across all phases.
    R: List[int] = [get_capacity(mn, mx, scarcity) for mn, mx in min_max]
    print(f"Capacities: {R}")
    total_resources = n_resources

    max_start_time = get_max_T(processes, max_phases, {r: R[r] for r in range(n_resources)}, k=100)

    def _zero_p(n_modes: int) -> List[int]:
        return [0] * n_modes

    def _zero_r(n_modes: int) -> List[List[int]]:
        return [[0] * total_resources for _ in range(n_modes)]

    job_idx = 1  # next job index to assign
    p: List[List[int]] = [_zero_p(1)]
    r: List[List[List[int]]] = [_zero_r(1)]
    M: List[int] = [1]
    ES: List[int] = [0]
    E: List[List[int]] = []  # precedence pairs [i, j]  (i before j)
    L: List[List[int]] = []  # linked-mode pairs
    O: List[int] = []  # last real job of each process

    for process in processes:
        n_active = len(process.network_type.value)
        phase_last: List[Optional[int]] = [None] * n_active

        for i in range(n_active):
            phase_tasks = process.tasks[i]
            n_tasks_in_phase = len(phase_tasks)
            # Modes are per-process per-phase: different processes may use
            # different RA-PSTs for the same phase index.
            M_i = process.phases[i].number_of_modes

            for j, task in enumerate(phase_tasks):
                # ---- precedence edges --------------------------------- #
                if j == 0:
                    preds = process.network_type.value[i]
                    if not preds:
                        E.append([0, job_idx])
                    else:
                        for pred_phase in preds:
                            if phase_last[pred_phase] is not None:
                                E.append([phase_last[pred_phase], job_idx])
                else:
                    E.append([job_idx - 1, job_idx])

                # ---- linked-mode pairs -------------------------------- #
                if j > 0:
                    L.append([job_idx - 1, job_idx])

                # ---- processing times & resource requirements --------- #
                p.append(_zero_p(M_i))
                r.append(_zero_r(M_i))
                M.append(M_i)
                for m in range(M_i):
                    p[job_idx][m] = task.duration[m]
                    res_idx: Optional[int] = task.resource[m]
                    if res_idx is not None:
                        r[job_idx][m][res_idx] = 1

                ES.append(process.start_time)

                if j == n_tasks_in_phase - 1:
                    phase_last[i] = job_idx

                job_idx += 1

        last_phase = n_active - 1
        if phase_last[last_phase] is not None:
            O.append(phase_last[last_phase])

    # Sink dummy job
    sink = job_idx
    p.append(_zero_p(1))
    r.append(_zero_r(1))
    M.append(1)
    ES.append(0)
    for last_job in O:
        E.append([last_job, sink])
    n = sink + 1

    # Transitive closure of precedence relations
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for i, j in E:
        adj[i].add(j)
    for k in range(n):
        for i in range(n):
            if k in adj[i]:
                adj[i] |= adj[k]
    TE = [[i, j] for i in range(n) for j in adj[i]]

    # Earliest start times
    for i in range(n):
        for k in range(n):
            if i in adj[k]:
                ES[i] = max(ES[i], ES[k] + min(p[k]))

    # Latest start times
    LS = [int(max_start_time)-1] * n
    for i in range(n-1, -1, -1):
        for k in range(n):
            if k in adj[i]:
                LS[i] = min(LS[i], LS[k] - min(p[i]))

    e_set = {(a, b) for a, b in TE} | {(b, a) for a, b in TE}
    VP = [
        [i, j]
        for i in range(n)
        for j in range(n)
        if i != j and (i, j) not in e_set
    ]

    return {
        "n": n,
        "T": int(max_start_time),
        "M": M,
        "R": R,
        "E": E,
        "L": L,
        "p": p,
        "r": r,
        "O": O,
        "ES": ES,
        "LS": LS,
        "VP": VP,
    }
