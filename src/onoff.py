import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from utils import get_earliest_start_time, get_latest_start_time

def onoff_model(n, T, M, R, E, p, L, r, VP, silent=True):
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
    model = gp.Model("onoff")
    if silent:
        model.setParam('OutputFlag', False)

    # Step variables
    step_sets = [(i, m, t) for i in range(n) for m in range(M) for t in range(T)]
    y = model.addVars(step_sets, vtype=GRB.BINARY, name="onoff")

    # Objective
    model.setObjective(gp.quicksum(t * y[n-1, m, t] for t in range(1,T) for m in range(M)), GRB.MINIMIZE)

    # Constraints
    # Schedule each job exactly once (schedule dummy separately, since p[n-1][m] = 0)
    model.addConstrs((gp.quicksum(y[i, m, t]/p[i][m] if p[i][m] > 0 else y[i, m, t] for m in range(M) for t in range(T) ) == 1 for i in range(1,n-1)), name="schedule")
    # model.addConstrs((gp.quicksum(y[i, m, t] for m in range(M) for t in range(T) if p[i][m]==0) == 1 for i in range(n)), name="schedule_dummy")
    model.addConstr((gp.quicksum(y[0, m, t] for m in range(M) for t in range(T)) == 1), name="schedule_first")
    model.addConstr((gp.quicksum(y[n-1, m, t] for m in range(M) for t in range(T)) == 1), name="schedule_last")

    # If job is started, start it for exactly p[i][m] consecutive time slots in mode m
    model.addConstrs((p[i][m] * (y[i, m, t] - y[i, m, t+1]) - gp.quicksum(y[i, m, tau] for tau in range(max(t - p[i][m]+1, 0), t)) <= 1 for i in range(n) for m in range(M) for t in range(T-1)), name="started_same_mode")

    # Precedence relations between jobs (i,j)
    model.addConstrs((gp.quicksum(y[i, m, tau]/max(p[i][m], 1) for m in range(M) for tau in range(t + 1 - min(p[i][m], 1))) >= gp.quicksum(y[j, m, t] for m in range(M)) for i,j in E for t in range(T)), name="precedence")

    # Resource availability
    model.addConstrs((gp.quicksum(r[i][m][k] * y[i, m, t] for i in range(n) for m in range(M)) <= R[k] for t in range(T) for k in range(len(R))), name="resource")

    # Linked modes of jobs (i,j)
    model.addConstrs((gp.quicksum(y[i, m, t]/max(p[i][m],1) for t in range(T)) == gp.quicksum(y[j, m, t]/max(p[j][m],1) for t in range(T)) for i,j in L for m in range(M)), name="linked")
    
    # Zero time slots 
    model.addConstrs((y[i, m, t] == 0 for i in range(n) for m in range(M) for t in range(earliest_starting_times[i])), name="zero_time_slots")
    
    return model, divisor

if __name__ == "__main__":
    # input = json.load(open("tests/simple.json"))
    input = json.load(open("tests/ra-pst.json"))
    start_time = time.time()
    model, divisor = onoff_model(n=input["n"], T=input["T"], M=input["M"], R=input["R"], E=input["E"], p=input["p"], L=input["L"], r=input["r"], VP=None)
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print("\033[91mModel infeasible\033[0m")
    else: 
        print("\033[92mModel feasible\033[0m")
        print("\033[93mModel does not handle processing time of 0 correctly\033[0m")
        print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
        for var in model.getVars():
            print(f"{var.varName}: {var.x}") if var.x > 0 else None
        print(f"Objective: {(model.objVal) * divisor}")

