from instances.generator import get_min_max_demands, get_capacity
from instances.definitions import ResourceLevel, Mode, NetworkType

from src.pulse import pulse_model


def get_or_instance(processes, scarcity, max_start_time, 
                    max_phases=3,
                    res_1_2_multiplier=2.0,
                    res_1_3_multiplier=3.0
                    ):
    """
    Translate the generated processes into an OR fiendly instance
    :param processes: List of processes
    :param scarcity: Resource scarcity
    :param max_start_time: Maximum start time of the processes
    :param max_phases: Maximum number of phases
    :param res_1_2_multiplier: Multiplier for resource 1 to resource 2 processing time
    :param res_1_3_multiplier: Multiplier for resource 1 to resource 3 processing time
    """
    print(f"Scarcity: {scarcity}")
    min_max_demands = get_min_max_demands(processes=processes, max_phases=max_phases) 
    min_max_demands[0][2] = (1, max(3,min_max_demands[0][2][1]))
    min_max_demands[1][2] = (1, max(3,min_max_demands[1][2][1]))
    min_max_demands[2][2] = (1, max(3,min_max_demands[2][2][1]))
    print(f"min_max_demands: {min_max_demands}")
    # min_max_demands[phase][resource]
    capacities = [[get_capacity(min_demand, max_demand, scarcity) for min_demand, max_demand in min_max_demands[i]] for i in range(max_phases)]
    # Translate to solver instance
    n = max_phases * 3 * len(processes) + 2 # #phases * #jobs_per_phase * #processes
    T = int(max_start_time*1.5)
    M = len(Mode)
    R = [capacities[i][j] for i in range(max_phases) for j in range(len(ResourceLevel))] # phase | resource
    print(f"Resource capacities: {R}")
    E = [] # nxn
    L = [] # nxn
    p = [[0 for _ in range(M)] for _ in range(n)] # (n+2)*M
    r = [[[0 for _ in range(len(ResourceLevel)*max_phases)] for _ in range(M)] for _ in range(n)] # n*M*R
    ES = [0 for _ in range(n)] # n
    # Populate E, L, p, r
    task_counter = 1
    for process in processes:
        for i, phase in enumerate(process.phases):
            if process.network_type == NetworkType.SINGLE and i >= 1: 
                task_counter += 3
                continue
            if process.network_type == NetworkType.DOUBLE and i >= 2: 
                task_counter += 3
                continue
            for j in range(3):
                # Precedence relations
                if i == 0 and j == 0:
                    E.append([0, task_counter])
                elif i == 1 and j == 0 and process.network_type == NetworkType.INTREE:
                    E.append([0, task_counter])
                elif i == 2 and j == 0 and process.network_type == NetworkType.INTREE:
                    E.append([task_counter-1, task_counter])
                    E.append([task_counter-4, task_counter])
                elif i == 2 and j == 2:
                    E.append([task_counter-1, task_counter])
                    E.append([task_counter, n-1])
                else:
                    E.append([task_counter-1, task_counter])
                if j == 2 and i == 1 and process.network_type == NetworkType.DOUBLE:
                    E.append([task_counter, n-1])
                if j == 2 and i == 0 and process.network_type == NetworkType.SINGLE:
                    E.append([task_counter, n-1])
                # Linked modes
                if j > 0:
                    L.append([task_counter-1, task_counter])
                # processing times and resource requirements
                for mode in Mode:
                    if j == 0 and mode == Mode.MODE_1:
                        p[task_counter][mode.value] = phase.resource_1_duration
                        r[task_counter][mode.value][i*max_phases + 0] = 1
                    elif j == 0 and mode == Mode.MODE_2:
                        p[task_counter][mode.value] = int(phase.resource_2_duration*res_1_2_multiplier)
                        r[task_counter][mode.value][i*max_phases + 1] = 1
                    elif j == 0 and mode == Mode.MODE_3:
                        p[task_counter][mode.value] = int(phase.resource_3_duration*res_1_3_multiplier)
                        r[task_counter][mode.value][i*max_phases + 2] = 1
                    elif j == 1 and mode == Mode.MODE_1:
                        p[task_counter][mode.value] = phase.resource_2_duration
                        r[task_counter][mode.value][i*max_phases + 1] = 1
                    elif j == 1 and mode == Mode.MODE_2:
                        p[task_counter][mode.value] = 0
                    elif j == 1 and mode == Mode.MODE_3:
                        p[task_counter][mode.value] = 0
                    elif j == 2 and mode == Mode.MODE_1:
                        p[task_counter][mode.value] = phase.resource_3_duration
                        r[task_counter][mode.value][i*max_phases + 2] = 1 
                    elif j == 2 and mode == Mode.MODE_2:
                        p[task_counter][mode.value] = phase.resource_3_duration
                        r[task_counter][mode.value][i*max_phases + 2] = 1
                    elif j == 2 and mode == Mode.MODE_3:
                        p[task_counter][mode.value] = 0
                ES[task_counter] = process.start_time
                task_counter += 1
    instance = {
        "n": n,
        "T": T,
        "M": M, 
        "R": R, 
        "E": E,
        "L": L,
        "p": p,
        "r": r,
        "ES": ES
    }
    return instance
