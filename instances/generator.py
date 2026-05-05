from __future__ import annotations

import random
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


def simulate_extremal(processes: List[Process], max_phases: int) -> List[List[PhaseTimeline]]:
    """
    For each resource index, build per-phase timelines using the mode that 
    maximises demand on that resource. 
    """
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
            modes = process.get_max_resource_demand_mode(resource)
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
# Extremal demand of phase timelines
# ---------------------------------------------------------------------------


def compute_min_demands(processes: List[Process], max_phases: int) -> List[Dict[int, int]]:
    """
    Kolisch et al. minimum demands:
    K_r^min = max_{j} { min_{m} { k_{jmr} } }

    :param processes: The processes containing the tasks
    :param max_phases: The maximum number of phases
    :return: A list of dictionaries of minimum demands for each phase
    """
    min_demands: List[Dict[int, int]] = [defaultdict(int) for _ in range(max_phases)]

    for process in processes:
        for phase_id, phase_tasks in enumerate(process.tasks):
            if phase_id >= max_phases: break
            for task in phase_tasks:
                if task is None: continue
                n_modes = len(task.resource)
                for res in {r for r in task.resource if r is not None}:
                    min_req = int(all(task.resource[m] == res for m in range(n_modes)))
                    min_demands[phase_id][res] = max(min_demands[phase_id][res], min_req)
    return [dict(d) for d in min_demands]



def get_extremal_demands(timelines: List[List[PhaseTimeline]]) -> List[Dict[int, int]]:
    """
    From a timeline of phases, return some extremal demand for each phase.
    :param timelines: List of phase timelines where jobs are scheduled at their earliest start time. 
    """
    n_phases = len(timelines[0])

    result: List[Dict[int, int]] = [defaultdict(int) for _ in range(n_phases)]

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
    processes: List[Process],
    max_phases: int = 3,
) -> List[List[Tuple[int, int]]]:
    """
    Get the minimum and maximum peak demand for each phase.
    Return ``demands[phase][resource_index] = (min, max)`` for every
    (phase, resource) pair.
    """
    timelines_max = simulate_extremal(processes, max_phases=max_phases)
    for idx, tl in enumerate(timelines_max, start=1):
        plot_timelines(tl, filename=f"timeline_max_{idx}.png")

    min_demands = compute_min_demands(processes, max_phases=max_phases)
    print("Min/max demands:", min_demands)
    max_demands = get_extremal_demands(timelines_max)
    print("Min/max demands:", max_demands)

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
