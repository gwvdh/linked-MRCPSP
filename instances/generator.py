from enum import Enum
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
from collections import defaultdict


class NetworkType(Enum):
    INTREE = "intree" # 1 -> 3, 2 -> 3
    SINGLE = "single" # 1 
    DOUBLE = "double" # 1 -> 3
    TRIPLE = "triple" # 1 -> 2 -> 3

class ResourceLevel(Enum):
    L1 = 0; L2 = 1; L3 = 2

class Mode(Enum):
    MODE_1 = 0 # L1(T1) -> L2(T1) -> L3(T1)
    MODE_2 = 1 # L2(T1+T2) -> L3(T3)
    MODE_3 = 2 # L3(T1+T2+T3)

@dataclass
class Task:
    duration: int
    resource: ResourceLevel
    earliest_start: int

class PhaseProfile:
    def __init__(self, base_duration: float, resource_1_ratio: float, resource_2_ratio: float):
        assert resource_1_ratio >= 1 and resource_2_ratio >= 1
        self.resource_3_duration = int(base_duration)
        self.resource_2_duration = int(self.resource_3_duration * resource_2_ratio)
        self.resource_1_duration = int(self.resource_2_duration * resource_1_ratio)
    
    def __str__(self):
        return f"({self.resource_1_duration}, {self.resource_2_duration}, {self.resource_3_duration})"


class Process:
    def __init__(self, network_type: NetworkType, phases: List[PhaseProfile]):
        self.network_type = network_type
        self.phases = phases

    def get_tasks(self, modes: List[Mode]) -> List[List[Task]]:
        """Get tasks for each phase based on the given modes"""
        phases = []
        for i, phase in enumerate(self.phases):
            if self.network_type == NetworkType.SINGLE and i >= 1: continue
            if self.network_type == NetworkType.DOUBLE and i >= 2: continue
            mode = modes[i]
            tasks = []

            # Find start time for the current phase
            start_time: float = 0
            if self.network_type == NetworkType.INTREE and (i == 0 or i == 1):
                start_time = 0
            elif self.network_type == NetworkType.INTREE and i == 2:
                start_time = max(sum(t.duration for t in phases[0]), sum(t.duration for t in phases[1]))
            else:
                start_time = sum(t.duration for phase_tasks in phases for t in phase_tasks)
            
            match mode:
                case Mode.MODE_1:   
                    tasks.append(Task(phase.resource_1_duration, ResourceLevel.L1, earliest_start=start_time+sum(t.duration for t in tasks)))
                    tasks.append(Task(phase.resource_2_duration, ResourceLevel.L2, earliest_start=start_time+sum(t.duration for t in tasks)))
                    tasks.append(Task(phase.resource_3_duration, ResourceLevel.L3, earliest_start=start_time+sum(t.duration for t in tasks)))
                case Mode.MODE_2:
                    tasks.append(Task(phase.resource_2_duration*2, ResourceLevel.L2, earliest_start=start_time+sum(t.duration for t in tasks)))
                    tasks.append(Task(phase.resource_3_duration, ResourceLevel.L3, earliest_start=start_time+sum(t.duration for t in tasks)))
                case Mode.MODE_3:
                    tasks.append(Task(phase.resource_3_duration*3, ResourceLevel.L3, earliest_start=start_time+sum(t.duration for t in tasks)))
                case _:
                    raise Exception("Invalid mode")
            phases.append(tasks)
        return phases

    def get_random_mode_tasks(self) -> List[List[Task]]:
        return self.get_tasks([random.choice([Mode.MODE_1, Mode.MODE_2, Mode.MODE_3]) for _ in range(len(self.phases))])
    
    def __str__(self):
        return f"network type: {self.network_type}\n Phases: " + " ".join([str(phase) for phase in self.phases])


def generate_instance(number_of_processes: int, arrival_rate: float, max_phases: int = 3):
    processes = []
    phase_profiles: List[PhaseProfile] = [PhaseProfile(random.uniform(1.0, 10.0), random.uniform(1.0, 3.0), random.uniform(1.0,3.0)) for _ in range(3)]
    for _ in range(number_of_processes):
        process_structure = random.choice([NetworkType.SINGLE, NetworkType.DOUBLE, NetworkType.TRIPLE, NetworkType.INTREE])
        print(f'Generating process with structure {process_structure}')
        # TODO: Do something with arrival times
        processes.append(Process(process_structure, phase_profiles))
    return processes


def simulate_processes(processes: List[Process], max_phases: int):
    # timeline[phase][time][resource] = count
    phase_timelines = [defaultdict(lambda: {r: 0 for r in ResourceLevel}) for _ in range(max_phases)]
    for process in processes:
        # Select random mode for each process
        process_tasks = process.get_random_mode_tasks()
        for i, phase_tasks in enumerate(process_tasks):
            for task in phase_tasks:
                for t in range(task.duration):
                    phase_timelines[i][task.earliest_start + t][task.resource] += 1
    return [dict(sorted(tl.items())) for tl in phase_timelines]


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_timelines(phase_timelines: List[Dict[int, Dict[ResourceLevel, int]]]):
    """
    Visualizes resource usage per phase using Stepped Line Charts.
    """
    # 1. Determine global time range
    all_times = [t for tl in phase_timelines for t in tl.keys()]
    if not all_times:
        print("No simulation data to plot.")
        return
    
    min_t, max_t = min(all_times), max(all_times)
    # Add a buffer to the end for the 'post' step visual
    time_range = list(range(min_t, max_t + 2)) 
    
    # 2. Setup Subplots (One per Phase)
    num_phases = len(phase_timelines)
    fig, axes = plt.subplots(num_phases, 1, figsize=(12, 4 * num_phases), sharex=True)
    if num_phases == 1: axes = [axes]

    # Distinct colors
    res_conf = {
        ResourceLevel.L1: {'color': '#1f77b4', 'label': 'Level 1'}, # Blue
        ResourceLevel.L2: {'color': '#ff7f0e', 'label': 'Level 2'}, # Orange
        ResourceLevel.L3: {'color': '#2ca02c', 'label': 'Level 3'}  # Green
    }

    for i, (ax, timeline) in enumerate(zip(axes, phase_timelines)):
        max_y = 0
        
        for res in [ResourceLevel.L1, ResourceLevel.L2, ResourceLevel.L3]:
            # Extract data, filling gaps with 0
            # Note: We append 0 at the end to close the step chart visually
            y_values = [timeline.get(t, {}).get(res, 0) for t in time_range[:-1]] + [0]
            max_y = max(max_y, max(y_values))
            
            # Plot Stepped Line
            ax.step(time_range, y_values, where='post', 
                    color=res_conf[res]['color'], 
                    label=res_conf[res]['label'], 
                    linewidth=2)
            
            # Light fill to show volume without obscuring other lines
            ax.fill_between(time_range, y_values, step='post', 
                            color=res_conf[res]['color'], 
                            alpha=0.1)

        ax.set_title(f"Phase {i+1} Resource Demand")
        ax.set_ylabel("Units Required")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, max_y + 1.5) # Add top margin
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Force integer ticks

        if i == 0: ax.legend(loc='upper right', framealpha=0.9)

    plt.xlabel("Time (Simulation Ticks)")
    plt.tight_layout()
    plt.show()


def monte_carlo_simulation(processes: List[Process], iterations: int, max_phases: int = 3):
    """Find the average resource demands for each time slot and phase by chosing random modes for each process."""
    avg_timelines = [defaultdict(lambda: {r: 0.0 for r in ResourceLevel}) for _ in range(max_phases)]
    
    print(f"Running {iterations} simulations...")
    for _ in range(iterations):
        single_run = simulate_processes(processes, max_phases)

        # Aggregate results
        for phase_id, timeline in enumerate(single_run):
            for t, res_counts in timeline.items():
                for res, count in res_counts.items():
                    avg_timelines[phase_id][t][res] += count

    # Divide by number of iterations to get average timelines
    for phase_id in range(max_phases):
        for t in avg_timelines[phase_id]:
            for res in ResourceLevel:
                avg_timelines[phase_id][t][res] /= iterations
                
    return [dict(sorted(tl.items())) for tl in avg_timelines]



if __name__ == "__main__":
    NUMBER_OF_PROCESSES = 20
    ARRIVAL_RATE = 1.0
    ITERATIONS = 1000

    print("Generating instances...")
    processes = generate_instance(number_of_processes=NUMBER_OF_PROCESSES, arrival_rate=ARRIVAL_RATE)

    print("Simulating (Arrival Rate = 0.5)...")
    results = monte_carlo_simulation(processes=processes, iterations=ITERATIONS)
    
    print("Simulation complete. Max time:", max(t for r in results for t in r.keys()) if results else 0)
    
    print("Plotting results...")
    plot_timelines(results)

