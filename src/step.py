import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from utils import get_earliest_start_time, get_latest_start_time

def step_model(n, T, M, R, E, p, L, r, VP, silent=True):
    """
    n: number of activities
    T: number of time slots 1,...,T
    M: number of modes
    R: List of resource capacities R[k]
    E: List of pairs of activity indices (i,j) indicating precedence relations
    p: List of processing times for each activity i in each mode m p[i][m]
    L: List of pairs of activity indices (i,j) indicating linked modes
    r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    """
    # Normalize processing times
    unique_processing_times = list(set([i for job in p for i in job]))
    unique_processing_times.append(T)
    divisor = gcd(*unique_processing_times)
    for i in range(len(p)):
        for j in range(len(p[i])):
            p[i][j] = p[i][j] // divisor
    T = T // divisor

    # Starting times
    earliest_starting_times = get_earliest_start_time(n, T, M, R, E, p, L, r, VP)
    latest_starting_times = get_latest_start_time(n, T, M, R, E, p, L, r, VP)

    # Initialize model
    model = gp.Model("step")
    if silent:
        model.setParam('OutputFlag', False)
    last_dummy = model.addVars(T, vtype=GRB.BINARY, name="last_dummy")

    # Step variables
    step_sets = [(i, m, t) for i in range(n) for m in range(M) for t in range(T)]
    z = model.addVars(step_sets, vtype=GRB.BINARY, name="pulse")

    # Objective
    model.setObjective(gp.quicksum(t * (z[n-1, m, t] - z[n-1, m, t-1]) for t in range(1,T) for m in range(M)), GRB.MINIMIZE)
    # Constraints
    # Schedule each job exactly once
    model.addConstrs((gp.quicksum(z[i, m, T-1] for m in range(M)) == 1 for i in range(n)), name="schedule")

    # If job is started, at or before $t-1$ in mode $m$, it has also started before $t$ in mode $m$
    model.addConstrs((z[i, m, t-1] <= z[i, m, t] for i in range(n) for m in range(M) for t in range(1, T)), name="started_same_mode")

    # Precedence relations between jobs (i,j)
    model.addConstrs((
        gp.quicksum((t + p[i][m]) * (z[i, m, t] - (z[i, m, t-1] if t-1>=0 else 0)) for t in range(T) for m in range(M)) <= 
        gp.quicksum(t * (z[j, m, t] - (z[j, m, t-1] if t-1>=0 else 0)) for t in range(T) for m in range(M))
        for i,j in E), 
        name="precedence")

    # Resource availability
    model.addConstrs((gp.quicksum(r[i][m][k] * (z[i, m, t] - (z[i, m, max(t - p[i][m], 0)] if t-p[i][m]>=0 else 0)) for m in range(M) for i in range(n)) <= R[k] 
                     for t in range(T) for k in range(len(R))), name="resource")

    # Linked modes of jobs (i,j)
    # model.addConstrs((z[i, 0, T-1] == 1 for i in range(n)), name="schedule")
    model.addConstrs((z[i, m, T-1] == z[j, m, T-1] for i,j in L for m in range(M)), name="linked")
    # model.addConstrs((gp.quicksum((z[i, m, t] - z[i, m, t-1]) for t in range(1,T)) <= gp.quicksum((z[j, m, t] - z[j, m, t-1]) for t in range(1,T)) 
    #                   for i,j in L for m in range(M)), name="linked")
    
    # Earliest start times
    model.addConstrs((z[i, m, t] == 0 for i in range(n) for m in range(M) for t in range(earliest_starting_times[i])), name="earliest_start_times")
    
    # Zero time slots (not needed, since release times and deadlines is a more specific setting)
    # model.addConstrs((z[i, m, 0] == 0 for i in range(n) for m in range(M)), name="zero_time_slots")
    
    return model, divisor

def step_model_disaggregated(n, T, M, R, E, p, L, r, VP, silent=True):
    """
    n: number of activities
    T: number of time slots 1,...,T
    M: number of modes
    R: List of resource capacities R[k]
    E: List of pairs of activity indices (i,j) indicating precedence relations
    p: List of processing times for each activity i in each mode m p[i][m]
    L: List of pairs of activity indices (i,j) indicating linked modes
    r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    """
    # Normalize processing times
    unique_processing_times = list(set([i for job in p for i in job]))
    unique_processing_times.append(T)
    divisor = gcd(*unique_processing_times)
    for i in range(len(p)):
        for j in range(len(p[i])):
            p[i][j] = p[i][j] // divisor
    T = T // divisor

    # Starting times
    earliest_starting_times = get_earliest_start_time(n, T, M, R, E, p, L, r, VP)
    latest_starting_times = get_latest_start_time(n, T, M, R, E, p, L, r, VP)

    # Initialize model
    model = gp.Model("step")
    if silent:
        model.setParam('OutputFlag', False)
    last_dummy = model.addVars(T, vtype=GRB.BINARY, name="last_dummy")

    # Step variables
    step_sets = [(i, m, t) for i in range(n) for m in range(M) for t in range(T)]
    z = model.addVars(step_sets, vtype=GRB.BINARY, name="pulse")

    # Objective
    model.setObjective(gp.quicksum(t * (z[n-1, m, t] - z[n-1, m, t-1]) for t in range(1,T) for m in range(M)), GRB.MINIMIZE)
    # Constraints
    # Schedule each job exactly once
    model.addConstrs((gp.quicksum(z[i, m, T-1] for m in range(M)) == 1 for i in range(n)), name="schedule")

    # If job is started, at or before $t-1$ in mode $m$, it has also started before $t$ in mode $m$
    model.addConstrs((z[i, m, t-1] <= z[i, m, t] for i in range(n) for m in range(M) for t in range(1, T)), name="started_same_mode")

    # Precedence relations between jobs (i,j)
    model.addConstrs((gp.quicksum(z[i, m, max(t - p[i][m], 0)] if t-p[i][m]>=0 else 0 for m in range(M)) >= gp.quicksum(z[j, m, t] for m in range(M)) for i,j in E for t in range(1,T)), name="precedence_disaggregated")
    # model.addConstrs((
        # gp.quicksum((t + p[i][m]) * (z[i, m, t] - z[i, m, t-1]) for m in range(M) for t in range(1,T)) <= 
        # gp.quicksum(t * (z[j, m, t] - z[j, m, t-1]) for m in range(M) for t in range(1,T))
        # for i,j in E), 
        # name="precedence")

    # Resource availability
    model.addConstrs((gp.quicksum(r[i][m][k] * (z[i, m, t] - (z[i, m, max(t - p[i][m], 0)] if t-p[i][m]>=0 else 0)) for m in range(M) for i in range(n)) <= R[k] 
                     for t in range(T) for k in range(len(R))), name="resource")

    # Linked modes of jobs (i,j)
    # model.addConstrs((z[i, 0, T-1] == 1 for i in range(n)), name="schedule")
    model.addConstrs((z[i, m, T-1] == z[j, m, T-1] for i,j in L for m in range(M)), name="linked")
    # model.addConstrs((gp.quicksum((z[i, m, t] - z[i, m, t-1]) for t in range(1,T)) <= gp.quicksum((z[j, m, t] - z[j, m, t-1]) for t in range(1,T)) 
    #                   for i,j in L for m in range(M)), name="linked")
    
    # Earliest start times
    model.addConstrs((z[i, m, t] == 0 for i in range(n) for m in range(M) for t in range(earliest_starting_times[i])), name="earliest_start_times")
    
    # Zero time slots (not needed, since release times and deadlines is a more specific setting)
    # model.addConstrs((z[i, m, 0] == 0 for i in range(n) for m in range(M)), name="zero_time_slots")
    
    return model, divisor

if __name__ == "__main__":
    input = json.load(open("tests/simple.json"))
    # input = json.load(open("tests/ra-pst.json"))
    start_time = time.time()
    model, divisor = step_model(n=input["n"], T=input["T"], M=input["M"], R=input["R"], E=input["E"], p=input["p"], L=input["L"], r=input["r"], VP=None)
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print("\033[91mModel infeasible\033[0m")
    else: 
        print("\033[92mModel feasible\033[0m")
        print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
        for var in model.getVars():
            print(f"{var.varName}: {var.x}") if var.x > 0 else None
        print(f"Objective: {(model.objVal) * divisor}")

