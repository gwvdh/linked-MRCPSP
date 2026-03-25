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
    min_max_demands = get_min_max_demands(processes=processes, max_phases=max_phases) 
    capacities = [[get_capacity(min_demand, max_demand, scarcity) for min_demand, max_demand in min_max_demands[i]] for i in range(max_phases)]
    print(f"Capacities: {capacities}")
    # Translate to solver instance
    n = 0 
    T = int(max_start_time)
    M = len(Mode)
    R = [capacities[i][j] for i in range(max_phases) for j in range(len(ResourceLevel))] # phase | resource
    E = [] # nxn
    L = [] # nxn
    p = []
    r = []
    ES = [] # n
    O = [] # O(n) Last jobs of each process
    # Populate E, L, p, r
    n += 1
    p.append([0 for _ in range(M)]) # (n+2)*M
    r.append([[0 for _ in range(len(ResourceLevel)*max_phases)] for _ in range(M)]) # n*M*R
    ES.append(0) # n
    for process in processes:
        for i, phase in enumerate(process.phases):
            if process.network_type == NetworkType.SINGLE and i >= 1: 
                continue
            if process.network_type == NetworkType.DOUBLE and i >= 2: 
                continue
            for j in range(3):
                # Precedence relations
                if i == 0 and j == 0:
                    E.append([0, n])
                elif i == 1 and j == 0 and process.network_type == NetworkType.INTREE:
                    E.append([0, n])
                elif i == 2 and j == 0 and process.network_type == NetworkType.INTREE:
                    E.append([n-1, n])
                    E.append([n-4, n])
                elif i == 2 and j == 2:
                    E.append([n-1, n])
                    O.append(n)
                else:
                    E.append([n-1, n])
                if j == 2 and i == 1 and process.network_type == NetworkType.DOUBLE:
                    O.append(n)
                if j == 2 and i == 0 and process.network_type == NetworkType.SINGLE:
                    O.append(n)
                # Linked modes
                if j > 0:
                    L.append([n-1, n])
                # processing times and resource requirements
                p.append([0 for _ in range(M)])
                r.append([[0 for _ in range(len(ResourceLevel)*max_phases)] for _ in range(M)])
                for mode in Mode:
                    p[n][mode.value] = process.tasks[i][j].duration[mode.value]
                    resource_demand = process.tasks[i][j].resource[mode.value]
                    for resource in ResourceLevel:
                        if resource == resource_demand:
                            r[n][mode.value][i*max_phases + resource.value] = 1
                            break
                ES.append(process.start_time)
                n += 1
    # Add last job
    n += 1
    p.append([0 for _ in range(M)]) # (n+2)*M
    r.append([[0 for _ in range(len(ResourceLevel)*max_phases)] for _ in range(M)]) # n*M*R
    ES.append(0) # n
    # Add last jobs of each process
    for job_index in O:
        E.append([job_index, n-1])

    # VP: List of pairs of activity indices (i,j) that are not precedence-related
    VP = [[i,j] for i in range(n) for j in range(n) if [i,j] not in E and [j,i] not in E and i != j]

    instance = {
        "n": n,
        "T": T,
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
    return instance
