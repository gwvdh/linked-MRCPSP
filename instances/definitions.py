from enum import Enum
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
import random
import sys


class NetworkType(Enum):
    INTREE = "intree" # 1 -> 3, 2 -> 3
    SINGLE = "single" # 1 
    DOUBLE = "double" # 1 -> 2
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
    def __init__(self, network_type: NetworkType, phases: List[PhaseProfile], start_time: int = 0, res_1_2_multiplier: float = 2.0, res_1_3_multiplier: float = 3.0):
        self.network_type = network_type
        self.phases = phases
        self.start_time = start_time
        self.resource_1_2_multiplier = res_1_2_multiplier
        self.resource_1_3_multiplier = res_1_3_multiplier

    def max_processing_time(self):
        return sum(phase.resource_1_duration + phase.resource_2_duration + phase.resource_3_duration for phase in self.phases)

    def get_min_resource_demand_mode(self, resource: ResourceLevel):
        """For each phase, select the mode with the minimum demand for the given resource"""
        # Mode for minimum demand: (mode, min_demand)
        phases = [(None, sys.maxsize)] * len(self.phases)
        for mode in Mode:
            tasks = self.get_tasks([mode] * len(self.phases))
            for phase_id, phase_tasks in enumerate(tasks):
                resource_tasks = [task for task in phase_tasks if task.resource == resource]
                resource_demand = sum(task.duration for task in resource_tasks)
                if resource_demand < phases[phase_id][1]:
                    phases[phase_id] = (mode, resource_demand)
        return self.get_tasks([phases[i][0] for i in range(len(self.phases))])

    def get_max_resource_demand_mode(self, resource: ResourceLevel):
        """For each phase, select the mode with the minimum demand for the given resource"""
        # Mode for minimum demand: (mode, min_demand)
        phases = [(None, 0)] * len(self.phases)
        for mode in Mode:
            tasks = self.get_tasks([mode] * len(self.phases))
            for phase_id, phase_tasks in enumerate(tasks):
                resource_tasks = [task for task in phase_tasks if task.resource == resource]
                resource_demand = sum(task.duration for task in resource_tasks)
                if resource_demand > phases[phase_id][1]:
                    phases[phase_id] = (mode, resource_demand)
        return self.get_tasks([phases[i][0] for i in range(len(self.phases))])

    def get_tasks(self, modes: List[Mode]) -> List[List[Task]]:
        """Get tasks for each phase based on the given modes"""
        phases = []
        for i, phase in enumerate(self.phases):
            if self.network_type == NetworkType.SINGLE and i >= 1: continue
            if self.network_type == NetworkType.DOUBLE and i >= 2: continue
            mode = modes[i]
            tasks = []

            # Find start time for the current phase
            start_time: float = self.start_time
            if self.network_type == NetworkType.INTREE and (i == 0 or i == 1):
                start_time = self.start_time
            elif self.network_type == NetworkType.INTREE and i == 2:
                start_time = self.start_time + max(sum(t.duration for t in phases[0]), sum(t.duration for t in phases[1]))
            else:
                start_time = self.start_time + sum(t.duration for phase_tasks in phases for t in phase_tasks)
            
            match mode:
                case Mode.MODE_1:   
                    tasks.append(Task(phase.resource_1_duration, ResourceLevel.L1, earliest_start=start_time+sum(t.duration for t in tasks)))
                    tasks.append(Task(phase.resource_2_duration, ResourceLevel.L2, earliest_start=start_time+sum(t.duration for t in tasks)))
                    tasks.append(Task(phase.resource_3_duration, ResourceLevel.L3, earliest_start=start_time+sum(t.duration for t in tasks)))
                case Mode.MODE_2:
                    tasks.append(Task(int(phase.resource_2_duration*self.resource_1_2_multiplier), ResourceLevel.L2, earliest_start=start_time+sum(t.duration for t in tasks)))
                    tasks.append(Task(phase.resource_3_duration, ResourceLevel.L3, earliest_start=start_time+sum(t.duration for t in tasks)))
                case Mode.MODE_3:
                    tasks.append(Task(int(phase.resource_3_duration*self.resource_1_3_multiplier), ResourceLevel.L3, earliest_start=start_time+sum(t.duration for t in tasks)))
                case _:
                    raise Exception("Invalid mode")
            phases.append(tasks)
        return phases

    def get_random_mode_tasks(self) -> List[List[Task]]:
        return self.get_tasks([random.choice([Mode.MODE_1, Mode.MODE_2, Mode.MODE_3]) for _ in range(len(self.phases))])
    
    def __str__(self):
        return f"network type: {self.network_type}\n Phases: " + " ".join([str(phase) for phase in self.phases])
