from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

if __name__ == "__main__":
    from vis import plot_combined_resource_demands
    from definitions import NetworkType, PhaseProfile, Process
    from xml_parser import RA_PST
else:
    from .vis import plot_combined_resource_demands
    from .definitions import NetworkType, PhaseProfile, Process
    from .xml_parser import RA_PST

XML_FILE = "rapst/full_rapst_permit.xml"
PhaseTimeline = Dict[int, Dict[int, int]]


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def generate_instance(
    number_of_processes: int,
    xml_files: Optional[List[Union[str, List[str]]]] = None,
    max_phases: int = 3,
    min_base_duration: float = 1.0,
    max_base_duration: float = 5.0,
    min_resource_ratio: float = 1.0,
    resource_ratio_center: float = 1.5,
    resource_ratio_spread: float = 1.0,
    arrival_rate: float = 0.7,
    batch_size: float = 3.0,
    verbose: bool = True,
) -> Tuple[List[Process], List[str]]:
    if xml_files is None:
        phase_pools = [[XML_FILE]] * max_phases
    else:
        raw = [xml_files[i % len(xml_files)] for i in range(max_phases)]
        phase_pools = [[e] if isinstance(e, str) else list(e) for e in raw]

    all_xml = sorted({f for pool in phase_pools for f in pool})
    ra_pst_cache = {f: RA_PST(f) for f in all_xml}
    global_resource_ids = sorted(
        {rid for rp in ra_pst_cache.values() for rid in rp.resource_ids}
    )
    for rp in ra_pst_cache.values():
        rp._global_resource_ids = global_resource_ids

    rng = np.random.default_rng()
    interarrival = rng.exponential(1.0 / arrival_rate, size=number_of_processes)
    batches = np.random.poisson(batch_size, number_of_processes)

    processes, start_time, i = [], 0, 0
    for period, batch in enumerate(batches):
        if period > 0:
            start_time += int(interarrival[period])
        for _ in range(int(batch)):
            if i >= number_of_processes:
                break
            phases = []
            for phase_idx in range(max_phases):
                ra_pst = ra_pst_cache[random.choice(phase_pools[phase_idx])]
                phases.append(PhaseProfile(
                    base_durations=[
                        random.uniform(min_base_duration, max_base_duration)
                        for _ in range(ra_pst.get_number_of_tasks())
                    ],
                    resource_ratios=[
                        max(min_resource_ratio,
                            np.random.normal(resource_ratio_center,
                                             resource_ratio_spread))
                        for _ in range(ra_pst.get_number_of_modes())
                    ],
                    ra_pst=ra_pst,
                ))
            structure = random.choice(list(NetworkType))
            if verbose:
                print(f"{i}: structure={structure.name}, start={start_time}")
            processes.append(
                Process(network_type=structure, phases=phases,
                        start_time=start_time)
            )
            i += 1
        if i >= number_of_processes:
            break

    return processes, global_resource_ids


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _active_phases(network_type: NetworkType) -> int:
    return len(network_type.value)


def _make_phase_timelines(max_phases: int) -> List[Dict[int, Dict[int, int]]]:
    return [defaultdict(lambda: defaultdict(int)) for _ in range(max_phases)]


def _collect_resource_indices(processes: List[Process]) -> List[int]:
    return sorted({
        res
        for process in processes
        for phase_tasks in process.tasks
        for task in phase_tasks
        if task is not None
        for res in task.resource
        if res is not None
    })


def _schedule_phase(
    process: Process,
    phase_index: int,
    mode: int,
    phase_end: List[int],
    phase_timelines: List[Dict],
) -> int:
    preds = process.network_type.value[phase_index]
    task_start = (
        process.start_time if not preds else max(phase_end[p] for p in preds)
    )
    for task in process.tasks[phase_index]:
        if task is None:
            continue
        dur, res = task.duration[mode], task.resource[mode]
        if res is not None:
            for t in range(dur):
                phase_timelines[phase_index][task_start + t][res] += 1
        task_start += dur
    return task_start


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------


def simulate_extremal(
    processes: List[Process], max_phases: int
) -> List[List[PhaseTimeline]]:
    timelines = []
    for resource in _collect_resource_indices(processes):
        phase_timelines = _make_phase_timelines(max_phases)
        for process in processes:
            modes = process.get_max_resource_demand_mode(resource)
            phase_end: List[int] = []
            for i in range(_active_phases(process.network_type)):
                if modes[i] is None:
                    phase_end.append(
                        phase_end[-1] if phase_end else process.start_time
                    )
                else:
                    phase_end.append(
                        _schedule_phase(process, i, modes[i],
                                        phase_end, phase_timelines)
                    )
        timelines.append(phase_timelines)
    return timelines


def simulate_processes(
    processes: List[Process], max_phases: int
) -> List[PhaseTimeline]:
    phase_timelines = _make_phase_timelines(max_phases)
    for process in processes:
        modes = [
            random.randrange(process.phases[i].number_of_modes)
            for i in range(len(process.phases))
        ]
        phase_end: List[int] = []
        for i in range(_active_phases(process.network_type)):
            phase_end.append(
                _schedule_phase(process, i, modes[i], phase_end, phase_timelines)
            )
    return [dict(sorted(tl.items())) for tl in phase_timelines]


# ---------------------------------------------------------------------------
# Demand computation
# ---------------------------------------------------------------------------


def compute_min_demands(
    processes: List[Process], max_phases: int
) -> List[Dict[int, int]]:
    min_demands = [defaultdict(int) for _ in range(max_phases)]
    for process in processes:
        for phase_id, phase_tasks in enumerate(process.tasks):
            if phase_id >= max_phases:
                break
            for task in phase_tasks:
                if task is None:
                    continue
                n_modes = len(task.resource)
                for res in {r for r in task.resource if r is not None}:
                    min_req = int(
                        all(task.resource[m] == res for m in range(n_modes))
                    )
                    min_demands[phase_id][res] = max(
                        min_demands[phase_id][res], min_req
                    )
    return [dict(d) for d in min_demands]


def get_extremal_demands(
    timelines: List[List[PhaseTimeline]],
) -> List[Dict[int, int]]:
    result = [defaultdict(int) for _ in range(len(timelines[0]))]
    for phase_timelines in timelines:
        for phase_id, phase_timeline in enumerate(phase_timelines):
            for res_counts in phase_timeline.values():
                for res, count in res_counts.items():
                    result[phase_id][res] = max(result[phase_id][res], count)
    return [dict(d) for d in result]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_min_max_demands(
    processes: List[Process], max_phases: int = 3
) -> List[Tuple[int, int]]:
    all_resource_indices = _collect_resource_indices(processes)
    timelines_max = simulate_extremal(processes, max_phases=max_phases)

    plot_combined_resource_demands(
        timelines_by_resource=timelines_max,
        resource_indices=all_resource_indices,
    )

    max_demands: Dict[int, int] = {}
    for r_idx, res_phase_timelines in zip(all_resource_indices, timelines_max):
        combined: Dict[int, int] = defaultdict(int)
        for phase_tl in res_phase_timelines:
            for t, res_counts in phase_tl.items():
                combined[t] += res_counts.get(r_idx, 0)
        max_demands[r_idx] = max(combined.values(), default=0)

    min_demands_by_phase = compute_min_demands(processes, max_phases)
    all_min_res = sorted({r for phase in min_demands_by_phase for r in phase})
    min_demands = {
        r: max(phase.get(r, 0) for phase in min_demands_by_phase)
        for r in all_min_res
    }

    print("Min demands:", {r: min_demands.get(r, 0) for r in all_resource_indices})
    print("Max demands:", max_demands)
    return [
        (min_demands.get(r, 0), max_demands.get(r, 0))
        for r in all_resource_indices
    ]


def get_capacity(min_demand: int, max_demand: int, scarcity: float) -> int:
    return int(min_demand + round(scarcity * (max_demand - min_demand)))


if __name__ == "__main__":
    procs, _ = generate_instance(number_of_processes=60, verbose=True)
    demands = get_min_max_demands(procs, max_phases=3)
    print("Min/max demands:", demands)
