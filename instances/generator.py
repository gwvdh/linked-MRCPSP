from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Optional

import numpy as np

from .definitions import NetworkType, PhaseProfile, Process
from .vis import plot_combined_resource_demands
from .xml_parser import RA_PST

DEFAULT_XML = "rapst/full_rapst_permit.xml"
PhaseTimeline = dict[int, dict[int, int]]


def generate_instance(
    number_of_processes: int,
    xml_files: Optional[list[str | list[str]]] = None,
    max_phases: int = 3,
    min_base_duration: float = 1.0,
    max_base_duration: float = 5.0,
    min_resource_ratio: float = 1.0,
    resource_ratio_center: float = 1.5,
    resource_ratio_spread: float = 1.0,
    arrival_rate: float = 0.7,
    batch_size: float = 3.0,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> tuple[list[Process], list[str]]:
    """
    Generate synthetic processes by sampling a phase template for each phase,
    drawing base task durations, and assigning a phase-network structure.
    """
    rng = np.random.default_rng(seed)

    if xml_files is None:
        phase_pools = [[DEFAULT_XML]] * max_phases
    else:
        raw = [xml_files[i % len(xml_files)] for i in range(max_phases)]
        phase_pools = [[x] if isinstance(x, str) else list(x) for x in raw]

    all_xml = sorted({path for pool in phase_pools for path in pool})
    ra_pst_by_file = {path: RA_PST(path) for path in all_xml}

    global_resource_ids = sorted(
        {rid for ra in ra_pst_by_file.values() for rid in ra.resource_ids}
    )
    for ra in ra_pst_by_file.values():
        ra._global_resource_ids = global_resource_ids

    resource_ratios = {
        rid: max(
            min_resource_ratio,
            float(rng.normal(resource_ratio_center, resource_ratio_spread)),
        )
        for rid in global_resource_ids
    }

    interarrival_times = rng.exponential(1.0 / arrival_rate, size=number_of_processes)
    batch_sizes = rng.poisson(batch_size, size=number_of_processes)

    processes: list[Process] = []
    current_start = 0

    for batch_idx, batch in enumerate(batch_sizes):
        if batch_idx > 0:
            current_start += int(interarrival_times[batch_idx])

        for _ in range(int(batch)):
            if len(processes) >= number_of_processes:
                break

            phases: list[PhaseProfile] = []
            for phase_idx in range(max_phases):
                ra_pst = ra_pst_by_file[rng.choice(phase_pools[phase_idx])]
                phases.append(
                    PhaseProfile(
                        base_durations=[
                            rng.uniform(min_base_duration, max_base_duration)
                            for _ in range(ra_pst.get_number_of_tasks())
                        ],
                        resource_ratios=resource_ratios,
                        ra_pst=ra_pst,
                    )
                )

            network = rng.choice(list(NetworkType))
            process = Process(
                network_type=network,
                phases=phases,
                start_time=current_start,
            )
            processes.append(process)

            if verbose:
                print(
                    f"{len(processes) - 1}: "
                    f"structure={network.name}, start={current_start}"
                )

        if len(processes) >= number_of_processes:
            break

    return processes, global_resource_ids


def active_phases(network_type: NetworkType) -> int:
    return len(network_type.value)


def make_phase_timelines(max_phases: int) -> list[DefaultDict[int, DefaultDict[int, int]]]:
    return [defaultdict(lambda: defaultdict(int)) for _ in range(max_phases)]


def collect_resource_indices(processes: list[Process]) -> list[int]:
    return sorted(
        {
            res
            for process in processes
            for phase_tasks in process.tasks
            for task in phase_tasks
            for res in task.resource
            if res is not None
        }
    )


def schedule_phase(
    process: Process,
    phase_idx: int,
    mode: int,
    phase_end: list[int],
    phase_timelines: list[DefaultDict[int, DefaultDict[int, int]]],
) -> int:
    predecessors = process.network_type.value[phase_idx]
    start = process.start_time if not predecessors else max(phase_end[p] for p in predecessors)

    t = start
    for task in process.tasks[phase_idx]:
        duration = task.duration[mode]
        resource = task.resource[mode]
        if resource is not None:
            for tau in range(duration):
                phase_timelines[phase_idx][t + tau][resource] += 1
        t += duration

    return t


def simulate_extremal(
    processes: list[Process],
    max_phases: int,
) -> list[list[PhaseTimeline]]:
    """
    For each resource, choose in every phase the mode with maximum demand for
    that resource and simulate earliest-start execution.
    """
    timelines_by_resource: list[list[PhaseTimeline]] = []

    for resource in collect_resource_indices(processes):
        phase_timelines = make_phase_timelines(max_phases)

        for process in processes:
            modes = process.get_max_resource_demand_mode(resource)
            phase_end: list[int] = []

            for phase_idx in range(active_phases(process.network_type)):
                mode = modes[phase_idx]
                if mode is None:
                    phase_end.append(phase_end[-1] if phase_end else process.start_time)
                    continue
                phase_end.append(
                    schedule_phase(process, phase_idx, mode, phase_end, phase_timelines)
                )

        timelines_by_resource.append(phase_timelines)

    return timelines_by_resource


def simulate_processes(
    processes: list[Process],
    max_phases: int,
    seed: Optional[int] = None,
) -> list[PhaseTimeline]:
    """
    Simulate processes under randomly selected phase modes.
    """
    rng = np.random.default_rng(seed)
    phase_timelines = make_phase_timelines(max_phases)

    for process in processes:
        modes = [
            int(rng.integers(process.phases[i].number_of_modes))
            for i in range(len(process.phases))
        ]

        phase_end: list[int] = []
        for phase_idx in range(active_phases(process.network_type)):
            phase_end.append(
                schedule_phase(
                    process,
                    phase_idx,
                    modes[phase_idx],
                    phase_end,
                    phase_timelines,
                )
            )

    return [dict(sorted(tl.items())) for tl in phase_timelines]


def compute_min_demands(
    processes: list[Process],
    max_phases: int,
) -> list[dict[int, int]]:
    """
    Minimum per-phase demand of each resource:
    a task contributes 1 to resource r iff it uses r in every mode.
    """
    min_demands = [defaultdict(int) for _ in range(max_phases)]

    for process in processes:
        for phase_idx, phase_tasks in enumerate(process.tasks):
            if phase_idx >= max_phases:
                break

            for task in phase_tasks:
                resources = [r for r in task.resource if r is not None]
                for res in set(resources):
                    mandatory = int(all(r == res for r in task.resource))
                    min_demands[phase_idx][res] = max(
                        min_demands[phase_idx][res],
                        mandatory,
                    )

    return [dict(d) for d in min_demands]


def get_extremal_demands(
    timelines_by_resource: list[list[PhaseTimeline]],
) -> list[dict[int, int]]:
    result = [defaultdict(int) for _ in range(len(timelines_by_resource[0]))]

    for resource_timelines in timelines_by_resource:
        for phase_idx, phase_timeline in enumerate(resource_timelines):
            for res_counts in phase_timeline.values():
                for res, count in res_counts.items():
                    result[phase_idx][res] = max(result[phase_idx][res], count)

    return [dict(d) for d in result]


def get_min_max_demands(
    processes: list[Process],
    max_phases: int = 3,
    plot: bool = True,
) -> list[tuple[int, int]]:
    """
    For each resource r, compute a pair:
      (minimum guaranteed demand, maximum observed extremal demand).
    """
    resource_indices = collect_resource_indices(processes)
    timelines_max = simulate_extremal(processes, max_phases=max_phases)

    if plot:
        plot_combined_resource_demands(
            timelines_by_resource=timelines_max,
            resource_indices=resource_indices,
        )

    max_demands: dict[int, int] = {}
    for res_idx, resource_phase_timelines in zip(resource_indices, timelines_max):
        combined = defaultdict(int)
        for phase_timeline in resource_phase_timelines:
            for t, counts in phase_timeline.items():
                combined[t] += counts.get(res_idx, 0)
        max_demands[res_idx] = max(combined.values(), default=0)

    min_demands_by_phase = compute_min_demands(processes, max_phases)
    all_min_resources = sorted({r for phase in min_demands_by_phase for r in phase})
    min_demands = {
        r: max(phase.get(r, 0) for phase in min_demands_by_phase)
        for r in all_min_resources
    }

    print("Min demands:", {r: min_demands.get(r, 0) for r in resource_indices})
    print("Max demands:", max_demands)

    return [(min_demands.get(r, 0), max_demands.get(r, 0)) for r in resource_indices]


def get_capacity(min_demand: int, max_demand: int, scarcity: float) -> int:
    return int(min_demand + round(scarcity * (max_demand - min_demand)))


if __name__ == "__main__":
    processes, resource_ids = generate_instance(number_of_processes=60, verbose=True)
    print(f"Resources: {resource_ids}")
    print(get_min_max_demands(processes, max_phases=3))
