from instances.generator import generate_instance, get_min_max_demands, get_capacity
from instances.definitions import ResourceLevel, Mode, NetworkType
from gurobipy import GRB
import time

from src.pulse import pulse_model
from src.vis_schedule import visualize_pulse_model


def get_or_instance(processes, scarcity, max_start_time, max_phases=3):
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
    T = int(max_start_time + 5)
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
                        p[task_counter][mode.value] = phase.resource_2_duration*2
                        r[task_counter][mode.value][i*max_phases + 1] = 1
                    elif j == 0 and mode == Mode.MODE_3:
                        p[task_counter][mode.value] = phase.resource_3_duration*3
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


def main():
    NUMBER_OF_PROCESSES = 30
    ARRIVAL_RATE = 0.7
    ITERATIONS = 50

    print("Generating instances...")
    processes = generate_instance(
        number_of_processes=NUMBER_OF_PROCESSES, 
        arrival_rate=ARRIVAL_RATE
    )

    max_start_time = max([p.start_time + p.max_processing_time() for p in processes])

    scarcities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for scarcity in scarcities:
        instance = get_or_instance(
            processes=processes, 
            scarcity=scarcity, 
            max_start_time=max_start_time,
            max_phases=3
        )
        # Solver
        start_time = time.time()
        model, divisor = pulse_model(n=instance["n"], T=instance["T"], M=instance["M"], R=instance["R"], E=instance["E"], p=instance["p"], L=instance["L"], r=instance["r"], ES=instance["ES"], VP=None)
        model.optimize()
        if model.status == GRB.INFEASIBLE:
            print("\033[91mModel infeasible\033[0m")
        else: 
            print("\033[92mModel feasible\033[0m")
            print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
            visualize_pulse_model(model, instance["n"], instance["T"], instance["M"], instance["R"], instance["p"], instance["r"], divisor, filename=f"Schedule_{scarcity}")
            for var in model.getVars():
                print(f"{var.varName}: {var.x}") if var.x > 0 else None
            print(f"Objective: {model.objVal * divisor}")
            

if __name__ == "__main__":
    main()



