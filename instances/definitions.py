from enum import Enum
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
import random
import sys
import numpy as np


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
    duration: List[int] # Duration of the task in each mode
    resource: List[ResourceLevel] # Resource level of the task in each mode
    start_time: int # Start time of the task

class PhaseProfile:
    def __init__(self, base_duration: float, resource_1_ratio: float, resource_2_ratio: float):
        assert resource_1_ratio >= 1 and resource_2_ratio >= 1
        self.resource_3_duration = int(base_duration)
        self.resource_2_duration = int(self.resource_3_duration * resource_2_ratio)
        self.resource_1_duration = int(self.resource_2_duration * resource_1_ratio)
    
    def __str__(self):
        return f"({self.resource_1_duration}, {self.resource_2_duration}, {self.resource_3_duration})"


class Process:
    def __init__(self, network_type: NetworkType, phases: List[PhaseProfile], start_time: int = 0, res_1_2_multiplier: float = 2.0, res_1_3_multiplier: float = 3.0, job_3_multiplier: float = 1.0):
        self.network_type = network_type
        self.phases = phases
        self.start_time = start_time
        self.resource_1_2_multiplier = res_1_2_multiplier
        self.resource_1_3_multiplier = res_1_3_multiplier
        self.job_3_multiplier = job_3_multiplier
        self.tasks = [[None for _ in range(3)] for _ in range(len(self.phases))] 
        self.define_tasks()

    def max_processing_time(self):
        return max(phase.resource_1_duration + phase.resource_2_duration + phase.resource_3_duration for phase in self.phases)*len(self.phases)

    def get_min_resource_demand_mode(self, resource: ResourceLevel):
        """For each phase, select the mode with the minimum demand for the given resource"""
        # Mode for minimum demand: (mode, min_demand)
        phases = [(None, sys.maxsize)] * len(self.phases)
        for mode in Mode:
            for phase_id, phase_tasks in enumerate(self.tasks):
                resource_tasks = [task for task in phase_tasks if task is not None and task.resource[mode.value] == resource]
                resource_demand = sum(task.duration[mode.value] for task in resource_tasks)
                if resource_demand < phases[phase_id][1]:
                    phases[phase_id] = (mode, resource_demand)
        # return the modes for each phase
        return [phase[0] for phase in phases]

    def get_max_resource_demand_mode(self, resource: ResourceLevel):
        """For each phase, select the mode with the minimum demand for the given resource"""
        # Mode for minimum demand: (mode, min_demand)
        phases = [(None, 0)] * len(self.phases)
        for mode in Mode:
            for phase_id, phase_tasks in enumerate(self.tasks):
                resource_tasks = [task for task in phase_tasks if task is not None and task.resource[mode.value] == resource]
                resource_demand = sum(task.duration[mode.value] for task in resource_tasks)
                if resource_demand > phases[phase_id][1]:
                    phases[phase_id] = (mode, resource_demand)
        return [phase[0] for phase in phases]

    def define_tasks(self, variance: float = 0.2) -> List[List[Task]]:
        for i, phase in enumerate(self.phases):
            if self.network_type == NetworkType.SINGLE and i >= 1: continue
            if self.network_type == NetworkType.DOUBLE and i >= 2: continue
            for j in range(3):
                if j == 0:
                    self.tasks[i][j] = Task(
                        [int(phase.resource_1_duration*np.random.normal(1.0, variance)),
                         int(phase.resource_2_duration*self.resource_1_2_multiplier*np.random.normal(1.0, variance)), 
                         int(phase.resource_3_duration*self.resource_1_3_multiplier*np.random.normal(1.0, variance))
                         ], 
                        [ResourceLevel.L1, ResourceLevel.L2, ResourceLevel.L3], 
                        start_time=self.start_time,
                    )
                elif j == 1:
                    self.tasks[i][j] = Task(
                        [int(phase.resource_2_duration*np.random.normal(1.0, variance)), 
                         0, 
                         0
                         ], 
                        [ResourceLevel.L2, None, None], 
                        start_time=self.start_time,
                    )
                elif j == 2:
                    self.tasks[i][j] = Task(
                        [int(phase.resource_3_duration*np.random.normal(1.0, variance)*self.job_3_multiplier), 
                         int(phase.resource_3_duration*np.random.normal(1.0, variance)*self.job_3_multiplier), 
                         0
                         ],
                        [ResourceLevel.L3, ResourceLevel.L3, None],
                        start_time=self.start_time,
                    )
                else:
                    raise Exception("Invalid task")
        return self.tasks

    def __str__(self):
        return f"network type: {self.network_type}\n Phases: " + " ".join([str(phase) for phase in self.phases])
