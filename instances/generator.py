import numpy as np
import random

if __name__ == "__main__":
    from vis import plot_timelines
    from definitions import *
else:
    from .vis import plot_timelines
    from .definitions import *


def generate_instance(number_of_processes: int, max_phases: int = 3,
                      min_base_duration: float = 1.0,
                      max_base_duration: float = 5.0,
                      min_resource_1_ratio: float = 1.0,
                      resource_1_ratio_center: float = 1.5,
                      resource_1_ratio_spread: float = 1.0,
                      min_resource_2_ratio: float = 1.0,
                      resource_2_ratio_center: float = 1.3,
                      resource_2_ratio_spread: float = 1.0,
                      arrival_rate: float = 0.7,
                      res_1_2_multiplier: float = 2.0,
                      res_1_3_multiplier: float = 3.0,
                      job_3_multiplier: float = 1.0
                      ):
    """
    Generates a list of processes with randomized phase profiles.
    :param number_of_processes: Number of processes to generate
    :param max_phases: Maximum number of phases per process
    :param min_base_duration: Minimum base duration of the phases
    :param max_base_duration: Maximum base duration of the phases
    :param min_resource_1_ratio: Minimum resource 1 ratio of the phases
    :param resource_1_ratio_center: Center of the resource 1 ratio normal distribution
    :param resource_1_ratio_spread: Spread of the resource 1 ratio normal distribution
    :param min_resource_2_ratio: Minimum resource 2 ratio of the phases
    :param resource_2_ratio_center: Center of the resource 2 ratio normal distribution
    :param resource_2_ratio_spread: Spread of the resource 2 ratio normal distribution
    :param arrival_rate: Arrival rate of the processes
    :param res_1_2_multiplier: Multiplier for resource 1 to resource 2 processing time
    :param res_1_3_multiplier: Multiplier for resource 1 to resource 3 processing time
    """
    processes = []
    phase_profiles: List[PhaseProfile] = []
    for _ in range(max_phases):
        base_duration = random.uniform(min_base_duration, max_base_duration)
        resource_1_ratio = max(
            min_resource_1_ratio, 
            np.random.normal(resource_1_ratio_center, resource_1_ratio_spread))
        resource_2_ratio = max(
            min_resource_2_ratio, 
            np.random.normal(resource_2_ratio_center, resource_2_ratio_spread))
        phase_profiles.append(PhaseProfile(base_duration, resource_1_ratio, resource_2_ratio))
    # Arrival times 
    start_time = 0
    max_periods = number_of_processes * 3
    interarrival_times = np.random.default_rng().exponential(
        scale=arrival_rate, size=max_periods)
    arrivals = np.random.poisson(arrival_rate, max_periods)
    i = 0
    period = 0
    while i < number_of_processes:
        if period > 0:
            start_time += int(interarrival_times[period])
        batch_size = arrivals[period]
        period += 1
        for j in range(batch_size):
            if i >= number_of_processes:
                break
            process_structure = random.choice([
                NetworkType.SINGLE, NetworkType.DOUBLE, 
                NetworkType.TRIPLE, NetworkType.INTREE])
            print(f'{i}: Generating process with structure {process_structure} and start time {start_time}')
            processes.append(Process(
                process_structure, 
                phase_profiles, 
                start_time=start_time,
                res_1_2_multiplier=res_1_2_multiplier,
                res_1_3_multiplier=res_1_3_multiplier,
                job_3_multiplier=job_3_multiplier
            ))
            i += 1
    return processes


def simulate_extermal(processes: List[Process], max_phases: int, get_min: bool = True):
    """Simulate for each resource the minimum demand"""
    resources = [ResourceLevel.L1, ResourceLevel.L2, ResourceLevel.L3]
    timelines = []
    for resource in resources:
        # Minimize demand for the current resource for each phase
        phase_timelines = [defaultdict(lambda: {r: 0 for r in ResourceLevel}) for _ in range(max_phases)]
        for process in processes:
            # Select random mode for each process
            if get_min:
                process_modes = process.get_min_resource_demand_mode(resource)
            else:
                process_modes = process.get_max_resource_demand_mode(resource)
            prev_task_start = process.start_time
            task_start = process.start_time
            for i, phase_tasks in enumerate(process.tasks):
                if process.network_type == NetworkType.INTREE and i == 1:
                    prev_task_start = task_start
                    task_start = process.start_time
                elif process.network_type == NetworkType.INTREE and i == 2:
                    task_start = max(prev_task_start, task_start)
                elif process.network_type == NetworkType.DOUBLE and i == 2: continue
                elif process.network_type == NetworkType.SINGLE and i >= 1: continue
                for task in phase_tasks:
                    if task is None or process_modes[i] is None: continue
                    for t in range(task.duration[process_modes[i].value]):
                        phase_timelines[i][task_start + t][task.resource[process_modes[i].value]] += 1
                    task_start += task.duration[process_modes[i].value]
        timelines.append(phase_timelines)
    return timelines


def get_extermal_demands(timeline: List[List[Dict[int, Dict[ResourceLevel, int]]]], get_min=True):
    """
    Get the maximum demand in the time line for each resource per phase
    :param timeline: List of [resource][phase] timelines, where resource is min/max demand for the phase.
    """
    extermal_demands = [[0 for _ in ResourceLevel] for _ in range(len(timeline))]
    if get_min:
        extermal_demands = [[100 for _ in ResourceLevel] for _ in range(len(timeline))]
    for resource_id, phase_timelines in enumerate(timeline):
        max_demands = [[0 for _ in ResourceLevel] for _ in range(len(timeline))]
        for phase_id, phase_timeline in enumerate(phase_timelines):
            for time, res_counts in phase_timeline.items():
                for res, count in res_counts.items():
                    #print(f"time = {time}, res = {res}, res.val = {res.value}, count = {count}")
                    #print(f"max: {max_demands[phase_id][res.value]}, count = {count}")
                    max_demands[phase_id][res.value] = max(max_demands[phase_id][res.value], count)
        for i in range(len(max_demands)):
            for j in range(len(max_demands[i])):
                if get_min:
                    extermal_demands[i][j] = min(extermal_demands[i][j], max_demands[i][j])
                else:
                    extermal_demands[i][j] = max(extermal_demands[i][j], max_demands[i][j])
        #print(f"max_demands = {max_demands}")
        #print(f"extermal_demands = {extermal_demands}")
        #input()
    return extermal_demands


def simulate_processes(processes: List[Process], max_phases: int):
    # timeline[phase][time][resource] = count
    phase_timelines = [defaultdict(lambda: {r: 0 for r in ResourceLevel}) for _ in range(max_phases)]
    for process in processes:
        # Select random mode for each process
        random_modes = [random.choice([Mode.MODE_1, Mode.MODE_2, Mode.MODE_3]) for _ in range(len(process.phases))]
        for i, phase_tasks in enumerate(process.tasks):
            for task in phase_tasks:
                for t in range(task.duration[random_modes[i]]):
                    phase_timelines[i][task.earliest_start + t][task.resource[random_modes[i]]] += 1
    return [dict(sorted(tl.items())) for tl in phase_timelines]


def monte_carlo_simulation(processes: List[Process], iterations: int, max_phases: int = 3):
    """Find the average resource demands for each time slot and phase by chosing random modes for each process."""
    avg_timelines = [defaultdict(lambda: {r: 0.0 for r in ResourceLevel}) for _ in range(max_phases)]
    max_timelines = [defaultdict(lambda: {r: 0.0 for r in ResourceLevel}) for _ in range(max_phases)]
    
    print(f"Running {iterations} simulations...")
    for _ in range(iterations):
        single_run = simulate_processes(processes, max_phases)

        # Aggregate results
        for phase_id, timeline in enumerate(single_run):
            for t, res_counts in timeline.items():
                for res, count in res_counts.items():
                    avg_timelines[phase_id][t][res] += count
                    max_timelines[phase_id][t][res] = max(max_timelines[phase_id][t][res], count)

    # Divide by number of iterations to get average timelines
    for phase_id in range(max_phases):
        for t in avg_timelines[phase_id]:
            for res in ResourceLevel:
                avg_timelines[phase_id][t][res] /= iterations
                
    return [dict(sorted(tl.items())) for tl in avg_timelines], [dict(sorted(tl.items())) for tl in max_timelines]

def get_min_max_demands(processes, max_phases=3):
    """
    Get the minimum and maximum demands for each resource per phase
    :param processes: List of processes
    :param max_phases: Maximum number of phases
    """
    resources = [ResourceLevel.L1, ResourceLevel.L2, ResourceLevel.L3]

    timelines_min = simulate_extermal(processes=processes, max_phases=max_phases, get_min=True)
    timelines_max = simulate_extermal(processes=processes, max_phases=max_phases, get_min=False)
    plot_timelines(timelines_min[0], filename="timeline_min_1.png")
    plot_timelines(timelines_max[0], filename="timeline_max_1.png")
    plot_timelines(timelines_min[1], filename="timeline_min_2.png")
    plot_timelines(timelines_max[1], filename="timeline_max_2.png")
    plot_timelines(timelines_min[2], filename="timeline_min_3.png")
    plot_timelines(timelines_max[2], filename="timeline_max_3.png")


    min_demands = get_extermal_demands(timelines_min, get_min=True)
    max_demands = get_extermal_demands(timelines_max, get_min=False)

    min_max_demands = [[(min_demands[i][r.value], max_demands[i][r.value]) for r in ResourceLevel] for i in range(max_phases)]
    return min_max_demands

def get_capacity(min_demand, max_demand, scarcity):
    return int(round(min_demand + scarcity * (max_demand - min_demand)))


if __name__ == "__main__":
    NUMBER_OF_PROCESSES = 50
    ARRIVAL_RATE = 0.7
    ITERATIONS = 50

    print("Generating instances...")
    processes = generate_instance(number_of_processes=NUMBER_OF_PROCESSES, arrival_rate=ARRIVAL_RATE)

    scarcities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for scarcity in scarcities:
        print(f"Scarcity = {scarcity}")
        min_max_demands = get_min_max_demands(processes=processes, max_phases=3)
        capacities = [[get_capacity(min_demand, max_demand, scarcity) for min_demand, max_demand in min_max_demands[i]] for i in range(3)]
        # Solver

    # print("Simulation complete. Max time:", max(t for r in avg_results for t in r.keys()) if avg_results else 0)
    
    # print("Plotting results...")
    # plot_timelines(avg_results, filename="timeline_avg.png")
    # plot_timelines(max_results, filename="timeline_max.png")

