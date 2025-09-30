import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from utils import get_earliest_start_time, get_latest_start_time

def continuous_model(n, T, M, R, E, VP, p, L, r, silent=True):
    """
    n: number of activities
    T: number of time slots 1,...,T
    M: List of for each activity i the number of modes M[i]
    R: List of resource capacities R[k]
    E: List of pairs of activity indices (i,j) indicating precedence relations
    VP: List of pairs of activity indices (i,j) that are not precedence-related
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
    model = gp.Model("continuous")
    if silent:
        model.setParam('OutputFlag', False)

    # Activity variables
    S = model.addVars(n, vtype=GRB.CONTINUOUS, name="activity")
    model.addConstrs((S[i] >= earliest_starting_times[i] for i in range(n)), name="earliest_starting_times")
    activity_mode_sets = [(i, m) for i in range(n) for m in range(M)]
    x = model.addVars(activity_mode_sets, vtype=GRB.BINARY, name="mode")
    pair_sets = [(i, j) for i in range(n) for j in range(n)]
    y = model.addVars(pair_sets, vtype=GRB.BINARY, name="completion_start_sequence")
    pair_sets_2 = [(i, j) for i in range(n) for j in range(n)]
    z = model.addVars(pair_sets_2, vtype=GRB.BINARY, name="start_start_sequence")
    resource_sets = [(i, j, k) for i in range(n) for j in range(n) for k in range(len(R))]
    u = model.addVars(resource_sets, vtype=GRB.BINARY, name="resource")

    # Objective
    model.setObjective(S[n-1], GRB.MINIMIZE)

    # Constraints
    # Connect the start-time variables with the completion-start sequencing variables
    model.addConstrs((S[i] + gp.quicksum(p[i][m] * x[i, m] for m in range(M)) <= S[j] + T*(1-y[i, j]) for i,j in VP), name="connect_start_completion")

    # Precedence relations between jobs (i,j)
    model.addConstrs((S[i] + gp.quicksum(p[i][m] * x[i, m] for m in range(M)) <= S[j] for i,j in E), name="connect_start_completion")

    # Connect the start-start sequencing variables and the timing variables
    model.addConstrs((T * z[i, j] >= S[j] - S[i] + 1 for i,j in VP), name="connect_start_start_timing")

    # Execute each activity in exactly one mode
    model.addConstrs((gp.quicksum(x[i, m] for m in range(M)) == 1 for i in range(n)), name="execute_activity")

    # Resource availability
    model.addConstrs((gp.quicksum(r[j][m][k] * x[j, m] for m in range(M)) - max(r[j][m][k] for m in range(M))*(1 - z[j, i] + y[j, i]) <= u[j,i,k] for j, i in VP for k in range(len(R))), name="resource")
    model.addConstrs((gp.quicksum(r[i][m][k] * x[i, m] for m in range(M)) + gp.quicksum(u[j, i, k] for j, i_prime in VP if i_prime == i) <= R[k] for i in range(n) for k in range(len(R))), name="resource_2")

    # Linked modes of jobs (i,j)
    model.addConstrs((x[i, m] == x[j, m] for i,j in L for m in range(M)), name="linked")
    
    # Zero time slots (not needed, since release times and deadlines is a more specific setting)
    
    return model, divisor

if __name__ == "__main__":
    # input = json.load(open("tests/simple.json"))
    input = json.load(open("tests/ra-pst-5.json"))
    start_time = time.time()
    model_infeasible = True
    VP = [[i,j] for i in range(input["n"]) for j in range(input["n"]) if [i,j] not in input["E"] and [j,i] not in input["E"] and i != j]
    model, divisor = continuous_model(n=input["n"], T=input["T"], M=input["M"], R=input["R"], E=input["E"], VP=VP, p=input["p"], L=input["L"], r=input["r"])
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print("\033[91mModel infeasible\033[0m")
    else: 
        print("\033[92mModel feasible\033[0m")
        print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
        for var in model.getVars():
            print(f"{var.varName}: {var.x}") if var.x > 0 else None
        print(f"Objective: {model.objVal * divisor}")

