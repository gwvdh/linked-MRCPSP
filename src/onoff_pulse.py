import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from utils import get_earliest_start_time, get_latest_start_time

def onoff_pulse_model(n, T, M, R, E, p, L, r, VP, silent=True):
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
    model = gp.Model("onoff_pulse")
    if silent:
        model.setParam('OutputFlag', False)

    # Pulse variables
    pulse_sets = [(i, m, t) for i in range(n) for m in range(M) for t in range(T)]
    x = model.addVars(pulse_sets, vtype=GRB.BINARY, name="pulse")
    y = model.addVars(pulse_sets, vtype=GRB.BINARY, name="onoff")

    # Objective
    model.setObjective(gp.quicksum(t * x[n-1, m, t] for t in range(T) for m in range(M)), GRB.MINIMIZE)

    # Constraints
    # Schedule job exactly once
    model.addConstrs((x.sum(i, "*", "*") == 1 for i in range(n)), name="schedule")

    # Connect pulse variables with onoff variables
    model.addConstrs((y[i, m, t] == gp.quicksum(x[i, m, tau] for tau in range(max(t - p[i][m] + 1, 0), t+1)) for i in range(n) for t in range(T) for m in range(M)), name="connect_pulse_onoff")
    model.addConstrs((gp.quicksum(y[i, m, t] for t in range(T)) == gp.quicksum(p[i][m] * x[i, m, t] for t in range(T)) for i in range(n) for m in range(M)), name="connect_pulse_onoff_processing_time")

    # Precedence relations between jobs (i,j)
    model.addConstrs((
        gp.quicksum((t + p[i][m]) * x[i, m, t] for m in range(M) for t in range(T)) <= 
        gp.quicksum(t * x[j, m, t] for m in range(M) for t in range(T))
        for i,j in E), 
        name="precedence")

    # Resource availability
    model.addConstrs((gp.quicksum(gp.quicksum(r[i][m][k] * x[i, m, tau] for tau in range(max(t-p[i][m]+1, 0), t+1)) for m in range(M) for i in range(n)) <= R[k] 
                     for t in range(T) for k in range(len(R))), name="resource")

    # Linked modes of jobs (i,j)
    model.addConstrs((gp.quicksum(x[i, m, t] for t in range(T)) <= gp.quicksum(x[j, m, t] for t in range(T)) 
                      for i,j in L for m in range(M)), name="linked")
    
    # Zero time slots 
    model.addConstrs((x[i, m, t] == 0 for i in range(n) for m in range(M) for t in range(earliest_starting_times[i])), name="zero_time_slots")
    
    return model, divisor

def onoff_pulse_model_disaggregated(n, T, M, R, E, p, L, r, VP, silent=True):
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
    model = gp.Model("onoff_pulse")
    if silent:
        model.setParam('OutputFlag', False)

    # Pulse variables
    pulse_sets = [(i, m, t) for i in range(n) for m in range(M) for t in range(T)]
    x = model.addVars(pulse_sets, vtype=GRB.BINARY, name="pulse")
    y = model.addVars(pulse_sets, vtype=GRB.BINARY, name="onoff")

    # Objective
    model.setObjective(gp.quicksum(t * x[n-1, m, t] for t in range(T) for m in range(M)), GRB.MINIMIZE)

    # Constraints
    # Schedule job exactly once
    model.addConstrs((x.sum(i, "*", "*") == 1 for i in range(n)), name="schedule")

    # Connect pulse variables with onoff variables
    model.addConstrs((y[i, m, t] == gp.quicksum(x[i, m, tau] for tau in range(max(t - p[i][m] + 1, 0), t+1)) for i in range(n) for t in range(T) for m in range(M)), name="connect_pulse_onoff")
    model.addConstrs((gp.quicksum(y[i, m, t] for t in range(T)) == gp.quicksum(p[i][m] * x[i, m, t] for t in range(T)) for i in range(n) for m in range(M)), name="connect_pulse_onoff_processing_time")

    # Precedence relations between jobs (i,j)
    model.addConstrs((
        gp.quicksum(x[i, m, tau] for m in range(M) for tau in range(t-p[i][m]+1)) >=
        gp.quicksum(x[j, m, tau] for m in range(M) for tau in range(t+1))
        for i,j in E for t in range(T)), 
        name="precedence")

    # Resource availability
    model.addConstrs((gp.quicksum(gp.quicksum(r[i][m][k] * x[i, m, tau] for tau in range(max(t-p[i][m]+1, 0), t+1)) for m in range(M) for i in range(n)) <= R[k] 
                     for t in range(T) for k in range(len(R))), name="resource")

    # Linked modes of jobs (i,j)
    model.addConstrs((gp.quicksum(x[i, m, t] for t in range(T)) <= gp.quicksum(x[j, m, t] for t in range(T)) 
                      for i,j in L for m in range(M)), name="linked")
    
    # Zero time slots 
    model.addConstrs((x[i, m, t] == 0 for i in range(n) for m in range(M) for t in range(earliest_starting_times[i])), name="zero_time_slots")
    
    return model, divisor

if __name__ == "__main__":
    # input = json.load(open("tests/simple.json"))
    input = json.load(open("tests/ra-pst.json"))
    start_time = time.time()
    model_infeasible = True
    model, x, divisor = onoff_pulse_model(n=input["n"], T=input["T"], M=input["M"], R=input["R"], E=input["E"], p=input["p"], L=input["L"], r=input["r"], VP=None)
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print("\033[91mModel infeasible\033[0m")
    else: 
        print("\033[92mModel feasible\033[0m")
        print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
        for key, val in model.getAttr("x", x).items():
            if val > 0:
                print(f"{key}: {val} \t p: {input['p'][int(key[0])][int(key[1])]}")
        print(f"Objective: {model.objVal * divisor}")
        
        

