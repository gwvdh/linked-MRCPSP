from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .generator import get_min_max_demands, get_capacity
from .definitions import Process, NetworkType
from .xml_parser import RA_PST


def get_or_instance(
    processes: List[Process],
    scarcity: float,
    max_start_time: int,
    ra_pst: RA_PST,
    max_phases: int = 3,
) -> Dict[str, Any]:
    """
    Translate generated processes into an OR instance. 

    Returns a dict with keys: n, T, M, R, E, L, p, r, O, ES, VP.
    """
    n_resources = ra_pst.get_number_of_resources()
    M = ra_pst.get_number_of_modes()

    min_max = get_min_max_demands(processes=processes, max_phases=max_phases)
    capacities: List[List[int]] = [
        [
            get_capacity(mn, mx, scarcity)
            for mn, mx in min_max[i]
        ]
        for i in range(max_phases)
    ]
    R: List[int] = [
        capacities[i][j]
        for i in range(max_phases)
        for j in range(n_resources)
    ]
    total_resources = max_phases * n_resources

    def _zero_p() -> List[int]:
        return [0] * M

    def _zero_r() -> List[List[int]]:
        return [[0] * total_resources for _ in range(M)]

    job_idx = 1          # next job index to assign
    p: List[List[int]] = [_zero_p()]
    r: List[List[List[int]]] = [_zero_r()]
    ES: List[int] = [0]
    E: List[List[int]] = []   # precedence pairs [i, j]  (i before j)
    L: List[List[int]] = []   # linked-mode pairs
    O: List[int] = []         # last real job of each process

    # phase_last_job[phase_index] = job index of last task in that phase
    # (used to wire inter-phase precedences)
    for process in processes:
        n_active = len(process.network_type.value)
        # first_job[phase] / last_job[phase] within this process
        phase_first: List[Optional[int]] = [None] * n_active
        phase_last: List[Optional[int]] = [None] * n_active

        for i in range(n_active):
            phase_tasks = process.tasks[i]
            n_tasks_in_phase = len(phase_tasks)

            for j, task in enumerate(phase_tasks):
                # ---- precedence edges --------------------------------- #
                if j == 0:
                    # First task of this phase
                    preds = process.network_type.value[i]

                    if not preds:
                        # No phase predecessor → connect from source
                        E.append([0, job_idx])
                    else:
                        for pred_phase in preds:
                            if phase_last[pred_phase] is not None:
                                E.append(
                                    [phase_last[pred_phase], job_idx]
                                )
                    phase_first[i] = job_idx
                else:
                    # Sequential within phase
                    E.append([job_idx - 1, job_idx])

                # ---- linked-mode pairs -------------------------------- #
                if j > 0:
                    L.append([job_idx - 1, job_idx])

                # ---- processing times & resource requirements --------- #
                p.append(_zero_p())
                r.append(_zero_r())
                for m in range(M):
                    p[job_idx][m] = task.duration[m]
                    res_idx: Optional[int] = task.resource[m]
                    if res_idx is not None:
                        # Offset by phase so each (phase, resource) pair
                        # maps to a unique column in r
                        col = i * n_resources + res_idx
                        r[job_idx][m][col] = 1

                ES.append(process.start_time)

                if j == n_tasks_in_phase - 1:
                    phase_last[i] = job_idx

                job_idx += 1

        # Last task of the final active phase → connect to sink later
        last_phase = n_active - 1
        if phase_last[last_phase] is not None:
            O.append(phase_last[last_phase])

    # Sink dummy job
    sink = job_idx
    p.append(_zero_p())
    r.append(_zero_r())
    ES.append(0)
    for last_job in O:
        E.append([last_job, sink])
    n = sink + 1

    e_set = {(a, b) for a, b in E} | {(b, a) for a, b in E}
    VP = [] 
    VP = [ # Inverse of precedence pairs. WARNING: it may be very large!
        [i, j]
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in e_set
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
        "VP": VP,
    }
