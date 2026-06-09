from ortools.sat.python import cp_model
import json
import time
import collections

if __name__ == "__main__":
    from utils import normalize
    import pickle
else:
    from .utils import normalize


def constraint_programming_model(n, T, M, R, E, p, L, r, O, VP=None, ES=None, LS=None, silent=True, obj="makespan", timeout=600):
    """
    Constraint programming model
    n: number of activities
    T: number of time slots 1,...,T
    M: number of modes for each activity
    R: List of resource capacities R[k]
    E: List of pairs of activity indices (i,j) indicating precedence relations
    p: List of processing times for each activity i in each mode m p[i][m]
    L: List of pairs of activity indices (i,j) indicating linked modes
    r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    O: List of last jobs indices of each process
    ES: Earliest start time for each activity i
    LS: Latest start time for each activity i
    """
    # Normalize processing times
    p, T, divisor = normalize(p, T)

    # Initialize model
    model = cp_model.CpModel()

    task_type = collections.namedtuple("task_type", "start end interval is_present")
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    all_tasks = {}
    resource_to_intervals = collections.defaultdict(list)

    # activities
    for i in range(n):
        present_vars = []
        for task_id, task in enumerate(range(M[i])):
            if sum(r[i][task]) != 1:
                resource = 0
            else:
                resource = r[i][task].index(1)
            duration = p[i][task]
            suffix = f"_{i}_{task_id}"
            start_var = model.new_int_var(ES[i], LS[i], "start" + suffix)
            end_var = model.new_int_var(ES[i], LS[i]+max(p[i]), "end" + suffix)
            is_present_var = model.new_bool_var(f"is_present{suffix}")
            present_vars.append(is_present_var)
            interval_var = model.new_optional_interval_var(start=start_var, size=duration, end=end_var, is_present=is_present_var, name="interval" + suffix)
            all_tasks[i, task_id] = task_type(
                start=start_var,
                end=end_var,
                interval=interval_var,
                is_present=is_present_var,
            )
            resource_to_intervals[resource].append(interval_var)
        # Present in exactly one mode
        model.add(sum(present_vars) == 1)

    # Resource overlap constraints
    for resource, intervals in resource_to_intervals.items():
        model.add_cumulative(
            intervals=intervals,
            demands=[1] * len(intervals),
            capacity=R[resource],
        )

    # Precedence constraints
    for i, j in E:
        for task_id in range(M[i]):
            for task_id2 in range(M[j]):
                model.add(
                    all_tasks[j, task_id2].start >= all_tasks[i, task_id].end
                ).only_enforce_if([all_tasks[i, task_id].is_present, all_tasks[j, task_id2].is_present])

    # Linked modes
    for i, j in L:
        for task_id in range(M[i]):
            model.add(all_tasks[i, task_id].is_present == all_tasks[j, task_id].is_present)

    completion_vars = []
    for i in range(n):
        completion = model.new_int_var(0, T, f"completion_{i}")
        completion_vars.append(completion)
        for task_id in range(M[i]):
            task = all_tasks[i, task_id]
            model.add(completion == task.end).only_enforce_if(task.is_present)

    if obj == "makespan":
        obj_var = model.new_int_var(0, T, "makespan")
        model.add_max_equality(obj_var, completion_vars)
        model.minimize(obj_var)
    elif obj == "flow-time":
        obj_var = model.new_int_var(0, T*n, "flow_time")
        model.add(obj_var == sum(completion_vars[i] - ES[i] for i in range(n)))
        model.minimize(obj_var)

    return model, divisor, obj_var


if __name__ == "__main__":
    input = json.load(open("data/78/instance_0.9.json"))
    # input = json.load(open("tests/ra-pst.json"))
    start_time = time.time()
    _model, _divisor, _obj_var = constraint_programming_model(n=input["n"], T=input["T"], M=input["M"], R=input["R"], E=input["E"], p=input["p"], L=input["L"], r=input["r"], ES=input["ES"], LS=input["LS"], O=None, 
                                                              obj="flow-time")
    _solver = cp_model.CpSolver()
    _solver.parameters.max_time_in_seconds = 600
    status = _solver.Solve(_model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\033[92mModel feasible\033[0m")
        print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
        print(f"Objective: {_solver.value(_obj_var) * _divisor}")
    else:
        print("\033[91mModel infeasible\033[0m")
    


