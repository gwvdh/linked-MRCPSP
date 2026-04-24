from __future__ import annotations

import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

if __name__ == "__main__":
    from vis import plot_timelines
    from definitions import NetworkType, PhaseProfile, Process
    from xml_parser import RA_PST
else:
    from .vis import plot_timelines
    from .definitions import NetworkType, PhaseProfile, Process
    from .xml_parser import RA_PST

XML_FILE = "rapst/full_rapst_permit.xml"

# Type alias: timeline[phase][time][resource_index] = count
PhaseTimeline = Dict[int, Dict[int, int]]


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def generate_instance(
    number_of_processes: int,
    max_phases: int = 3,
    min_base_duration: float = 1.0,
    max_base_duration: float = 5.0,
    min_resource_ratio: float = 1.0,
    resource_ratio_center: float = 1.5,
    resource_ratio_spread: float = 1.0,
    arrival_rate: float = 0.7,
    batch_size: float = 3.0,
    verbose: bool = True,
) -> List[Process]:
    """
    Generate a list of processes with randomised phase profiles.

    :param number_of_processes: Total number of processes to generate
    :param max_phases: Number of phase profiles to create (shared across processes)
    :param min_base_duration: Lower bound for task base durations
    :param max_base_duration: Upper bound for task base durations
    :param min_resource_ratio: Floor applied to sampled resource ratios
    :param resource_ratio_center: Mean of the resource-ratio normal distribution
    :param resource_ratio_spread: Std-dev of the resource-ratio normal distribution
    :param arrival_rate: Rate λ of the Poisson arrival process
    :param batch_size: Mean batch size (Poisson-distributed)
    :param verbose: Print per-process summary
    """
    ra_pst = RA_PST(XML_FILE)
    n_tasks = ra_pst.get_number_of_tasks()
    n_modes = ra_pst.get_number_of_modes()

    # Generate phase profiles
    phase_profiles: List[PhaseProfile] = []
    for _ in range(max_phases):
        base_durations = [
            random.uniform(min_base_duration, max_base_duration)
            for _ in range(n_tasks)
        ]
        resource_ratios = [
            max(
                min_resource_ratio,
                np.random.normal(resource_ratio_center, resource_ratio_spread),
            )
            for _ in range(n_modes)
        ]
        phase_profiles.append(
            PhaseProfile(
                base_durations=base_durations,
                resource_ratios=resource_ratios,
                ra_pst=ra_pst,
            )
        )

    rng = np.random.default_rng()
    interarrival_times = rng.exponential(
        scale=1.0 / arrival_rate, size=number_of_processes
    )
    batch_sizes = np.random.poisson(batch_size, number_of_processes)
    # Generate processes
    processes: List[Process] = []
    start_time = 0
    period = 0
    i = 0
    while i < number_of_processes:
        if period > 0:
            start_time += int(interarrival_times[period])
        _batch = int(batch_sizes[period])
        period += 1
        for _ in range(_batch):
            if i >= number_of_processes:
                break
            structure = random.choice(list(NetworkType))
            if verbose:
                print(f"{i}: structure={structure.name}, start={start_time}")
            processes.append(
                Process(
                    network_type=structure,
                    phases=phase_profiles,
                    start_time=start_time,
                )
            )
            i += 1

    return processes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _active_phases(network_type: NetworkType) -> int:
    """Number of phases actually executed for a given network type."""
    return len(network_type.value)


def _make_phase_timelines(max_phases: int) -> List[Dict[int, Dict[int, int]]]:
    """
    Create per-phase timeline structures.
    Outer dict: time → inner dict.
    Inner dict: resource index (int) → count, defaulting to 0.
    """
    return [defaultdict(lambda: defaultdict(int)) for _ in range(max_phases)]


def _schedule_phase(
    process: Process,
    phase_index: int,
    mode: int,
    phase_end: List[int],
    phase_timelines: List[Dict],
) -> int:
    """
    Walk all tasks in one phase, accumulate resource usage into 
    phase_timelines, and return the time at which the phase ends.
    """
    preds = process.network_type.value[phase_index]
    task_start = (
        process.start_time
        if not preds
        else max(phase_end[p] for p in preds)
    )

    for task in process.tasks[phase_index]:
        if task is None:
            continue
        dur = task.duration[mode]
        res: Optional[int] = task.resource[mode]
        if res is not None:
            for t in range(dur):
                phase_timelines[phase_index][task_start + t][res] += 1
        task_start += dur

    return task_start


# ---------------------------------------------------------------------------
# Extremal simulation
# ---------------------------------------------------------------------------


def simulate_extremal(
    processes: List[Process],
    max_phases: int,
    get_min: bool = True,
) -> List[List[PhaseTimeline]]:
    """
    For each resource index, build per-phase timelines using the mode that
    minimises (get_min=True) or maximises (get_min=False) demand on that
    resource.

    Iterates over plain resource indices (int) so comparisons against
    ``task.resource[mode]`` (also int) are always valid.

    Returns ``timelines[resource_index][phase_index][time][resource_index]``.
    """
    # Derive the set of resource indices from the first process's tasks.
    # All processes share the same RA-PST so the resource set is identical.
    all_resource_indices: List[int] = sorted(
        {
            res
            for phase_tasks in processes[0].tasks
            for task in phase_tasks
            if task is not None
            for res in task.resource
            if res is not None
        }
    )

    timelines: List[List[PhaseTimeline]] = []

    for resource in all_resource_indices:
        phase_timelines = _make_phase_timelines(max_phases)

        for process in processes:
            # get_min/max_resource_demand_mode now receives a plain int
            modes: List[Optional[int]] = (
                process.get_min_resource_demand_mode(resource)
                if get_min
                else process.get_max_resource_demand_mode(resource)
            )
            n_active = _active_phases(process.network_type)
            phase_end: List[int] = []

            for i in range(n_active):
                mode = modes[i]
                if mode is None:
                    phase_end.append(
                        phase_end[-1] if phase_end else process.start_time
                    )
                    continue
                end = _schedule_phase(process, i, mode, phase_end, phase_timelines)
                phase_end.append(end)

        timelines.append(phase_timelines)

    return timelines


# ---------------------------------------------------------------------------
# Random-mode simulation
# ---------------------------------------------------------------------------


def simulate_processes(
    processes: List[Process],
    max_phases: int,
) -> List[PhaseTimeline]:
    """
    Simulate one run by picking a random mode per phase per process.
    Returns ``phase_timelines[phase_index][time][resource_index]``.
    """
    phase_timelines = _make_phase_timelines(max_phases)

    for process in processes:
        n_active = _active_phases(process.network_type)
        random_modes: List[int] = [
            random.randrange(process.phases[i].number_of_modes)
            for i in range(len(process.phases))
        ]
        phase_end: List[int] = []

        for i in range(n_active):
            mode = random_modes[i]
            end = _schedule_phase(process, i, mode, phase_end, phase_timelines)
            phase_end.append(end)

    return [dict(sorted(tl.items())) for tl in phase_timelines]


# ---------------------------------------------------------------------------
# Monte-Carlo aggregation
# ---------------------------------------------------------------------------


def monte_carlo_simulation(
    processes: List[Process],
    iterations: int,
    max_phases: int = 3,
) -> Tuple[List[PhaseTimeline], List[PhaseTimeline]]:
    """
    Simulate one run by picking a random mode per phase per process.
    """
    avg_timelines: List[Dict] = [
        defaultdict(lambda: defaultdict(float)) for _ in range(max_phases)
    ]
    max_timelines: List[Dict] = [
        defaultdict(lambda: defaultdict(float)) for _ in range(max_phases)
    ]

    print(f"Running {iterations} simulations...")
    for _ in range(iterations):
        single_run = simulate_processes(processes, max_phases)
        for phase_id, timeline in enumerate(single_run):
            for t, res_counts in timeline.items():
                for res, count in res_counts.items():
                    avg_timelines[phase_id][t][res] += count
                    if count > max_timelines[phase_id][t][res]:
                        max_timelines[phase_id][t][res] = count

    for phase_id in range(max_phases):
        for t in avg_timelines[phase_id]:
            for res in avg_timelines[phase_id][t]:
                avg_timelines[phase_id][t][res] /= iterations

    return (
        [dict(sorted(tl.items())) for tl in avg_timelines],
        [dict(sorted(tl.items())) for tl in max_timelines],
    )


# ---------------------------------------------------------------------------
# Extremal demand of phase timelines
# ---------------------------------------------------------------------------


def get_extremal_demands(
    timelines: List[List[PhaseTimeline]],
    get_min: bool = True,
) -> List[Dict[int, int]]:
    """
    From a timeline of phases, return some extram demand for each phase.
    :param timelines: List of phase timelines where jobs are scheduled at their earliest start time. 
    :param get_min: Get the minimum or maximum peak demand
    """
    n_phases = len(timelines[0])
    sentinel = sys.maxsize if get_min else 0

    extremal: List[Dict[int, int]] = [
        defaultdict(lambda: sentinel) for _ in range(n_phases)
    ]

    for _resource_id, phase_timelines in enumerate(timelines):
        peaks: List[Dict[int, int]] = [defaultdict(int) for _ in range(n_phases)]

        for phase_id, phase_timeline in enumerate(phase_timelines):
            for _time, res_counts in phase_timeline.items():
                for res, count in res_counts.items():
                    if count > peaks[phase_id][res]:
                        peaks[phase_id][res] = count

        for i in range(n_phases):
            for res, peak in peaks[i].items():
                if get_min:
                    extremal[i][res] = min(extremal[i][res], peak)
                else:
                    extremal[i][res] = max(extremal[i][res], peak)

    return [dict(phase) for phase in extremal]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_min_max_demands(
    processes: List[Process],
    max_phases: int = 3,
) -> List[List[Tuple[int, int]]]:
    """
    Get the minimum and maximum peak demand for each phase.
    """
    timelines_min = simulate_extremal(processes, max_phases=max_phases, get_min=True)
    timelines_max = simulate_extremal(processes, max_phases=max_phases, get_min=False)

    for idx, (tl_min, tl_max) in enumerate(
        zip(timelines_min, timelines_max), start=1
    ):
        plot_timelines(tl_min, filename=f"timeline_min_{idx}.png")
        plot_timelines(tl_max, filename=f"timeline_max_{idx}.png")

    min_demands = get_extremal_demands(timelines_min, get_min=True)
    # NOTE: override computed minimums with a known safe floor if desired:
    # min_demands = [{r: 0 for r in range(n_resources)} for _ in range(max_phases)]
    max_demands = get_extremal_demands(timelines_max, get_min=False)

    all_resources = sorted(
        {res for phase in (*min_demands, *max_demands) for res in phase}
    )

    return [
        [
            (min_demands[i].get(res, 0), max_demands[i].get(res, 0))
            for res in all_resources
        ]
        for i in range(max_phases)
    ]


def get_capacity(min_demand: int, max_demand: int, scarcity: float) -> int:
    """Interpolate between min and max demand by a scarcity factor ∈ [0, 1]."""
    return int(min_demand + round(scarcity * (max_demand - min_demand)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    procs = generate_instance(number_of_processes=60, verbose=True)
    demands = get_min_max_demands(procs, max_phases=3)
    print("Min/max demands:", demands)
