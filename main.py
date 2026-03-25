from instances.generator import generate_instance, get_min_max_demands, get_capacity
from instances.definitions import ResourceLevel, Mode, NetworkType
from instances.or_instance import get_or_instance
from gurobipy import GRB
import gurobipy as gp
import time
import copy

from src.pulse import pulse_model, pulse_model_disaggregated
from src.step import step_model, step_model_disaggregated
from src.onoff import onoff_model
from src.onoff_pulse import onoff_pulse_model, onoff_pulse_model_disaggregated
from src.continuous import continuous_model
from src.vis_schedule import visualize_pulse_model
from database import Database
import json
import os
import datetime


def model_selector(model: str, n, T, M, R, E, p, L, r, O, ES, VP, obj="makespan", timeout=600):
    if model == "pulse" or model == "PDT":
        return pulse_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "pulse-disaggregated" or model == "PDDT":
        return pulse_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "step" or model == "SDT":
        return step_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "step-disaggregated" or model == "SDDT":
        return step_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "onoff" or model == "OODDT":
        return onoff_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "onoff-pulse" or model == "OOPDT":
        return onoff_pulse_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "onoff-pulse-disaggregated" or model == "OOPDDT":
        return onoff_pulse_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout)
    if model == "continuous" or model == "MSEQCT":
        return continuous_model(n=n, T=T, M=M, R=R, E=E, ES=ES, VP=VP, p=p, L=L, r=r, O=O, obj=obj, timeout=timeout)
    raise ValueError(f"Unknown model {model}")


def output_instance(instance, scarcity, max_start_time, max_phases, res_1_2_multiplier, res_1_3_multiplier, folder_name=""):
    if not os.path.exists(f"data/{folder_name}"):
        os.mkdir(f"data/{folder_name}")
    filename = f"data/{folder_name}/instance_{scarcity}_{max_start_time}_{max_phases}_{res_1_2_multiplier}_{res_1_3_multiplier}.json"
    with open(filename, "w") as f:
        json.dump(instance, f)
    return filename

def output_solution(model, model_name, scarcity, max_start_time, max_phases, res_1_2_multiplier, res_1_3_multiplier, folder_name=""):
    if not os.path.exists(f"data/{folder_name}"):
        os.mkdir(f"data/{folder_name}")
    sol_file = f"data/{folder_name}/solution_{model_name}_{scarcity}.json"
    model.write(sol_file)
    return sol_file


def create_tables(instance_id, db, calculate_unknown=False):
    models = ["PDT", "PDDT", "SDT", "SDDT", "OODDT", "OOPDT", "OOPDDT", "MSEQCT"]
    scarcities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    instance = db.get_instance(instance_id)
    print(f"Instance: {instance}")
    latex_table = ""
    latex_table += "\\begin{table}[ht]\n"
    latex_table += "\\centering\n"
    latex_time_table = copy.deepcopy(latex_table)
    latex_table += "\\caption{Makespan (" + f"{instance[1]}" + " processes). An entry with - indicates an infeasible instance.}\n"
    latex_table += "\\label{tab:makespan}\n"
    latex_table += "\\begin{tabular}{|l|" + "c|" * len(scarcities) + "}\n"
    latex_table += "\\hline\n"
    latex_table += "\(RS\) &" + " & ".join([f"{scarcity:.1f}" for scarcity in scarcities]) + " \\\\ \\hline\n"
    latex_time_table += "\\caption{Runtime (" + f"{instance[1]}" + " processes). An entry with - indicates an infeasible instance.}\n"
    latex_time_table += "\\label{tab:runtime}\n"
    latex_time_table += "\\begin{tabular}{|l|" + "c|" * len(scarcities) + "}\n"
    latex_time_table += "\\hline\n"
    latex_time_table += "\(RS\) &" + " & ".join([f"{scarcity:.1f}" for scarcity in scarcities]) + " \\\\ \\hline\n"
    for model_name in models:
        latex_table += f"{model_name}"
        latex_time_table += f"{model_name}"
        for scarcity in scarcities:
            solution = db.get_solution(instance_id=instance_id, model_name=model_name, scarcity=scarcity)
            print(f"Model: {model_name}\tScarcity: {scarcity}\tSolution: {solution}")
            if solution is None:
                print("\033[91mSolution not found\033[0m")
                latex_table += " & ?"
                latex_time_table += " & ?"
                continue
            #model = gp.read(f'{solution[3][:-4]}.bas')

            solution_json = json.load(open(f'{solution[3]}'))
            if solution_json['SolutionInfo']['Status'] == GRB.INFEASIBLE:
                print("\033[91mModel infeasible\033[0m")
                latex_table += " & -"
                latex_time_table += " & -"
            elif solution_json['SolutionInfo']['SolCount'] == 0 and solution_json['SolutionInfo']['Status'] == GRB.TIME_LIMIT:
                print("\033[91mModel solution not found\033[0m")
                latex_table += " & -"
                latex_time_table += " & -"
            else: 
                print("\033[92mModel feasible\033[0m")
                #if model_name == "PDT" or model_name == "PDDT" or model_name == "OOPDT" or model_name == "OOPDDT":
                #    visualize_pulse_model(model=model, 
                #                        n=instance["n"], 
                #                        T=instance["T"], 
                #                        M=instance["M"], 
                #                        R=instance["R"], 
                #                        p=instance["p"], 
                #                        r=instance["r"], 
                #                        processes=processes,
                #                        divisor=divisor, 
                #                        filename=f"Schedule_{scarcity}"
                #                        )
                #elif model_name == "SDT" or model_name == "SDDT":
                #    pass
                #elif model_name == "OODDT":
                #    pass
                #elif model_name == "MSEQCT":
                #    pass
                #for var in model.getVars():
                #    pass
                #    #print(f"{var.varName}: {var.x}") if var.x > 0 else None
                objval = solution_json['SolutionInfo']['ObjVal']
                print(f"Objective: {objval}")
                if solution_json['SolutionInfo']['Status'] == GRB.OPTIMAL:
                    latex_table += " & " + f"\\textbf{{{objval:.1f}}}"
                else:
                    latex_table += " & " + f"{objval:.1f}"
                latex_time_table += f" & {solution_json['SolutionInfo']['Runtime']:.2f}s"
        latex_table += " \\\\ \\hline\n"
        latex_time_table += " \\\\ \\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    latex_time_table += "\\end{tabular}\n"
    latex_time_table += "\\end{table}"
    print(latex_table)
    print(latex_time_table)


def test_model(processes, solver, scarcity, objective, MAX_PHASES=3, RES_1_2_MULTIPLIER=2.0, RES_1_3_MULTIPLIER=3.0, timeout=600, db=None, db_instance_id=None):
    if db is None:
        db = Database("database.db")
    if db_instance_id is None:
        raise Exception("No instance id provided")
    print(f"Model: {solver}\tScarcity: {scarcity}")
    max_start_time = int(max([p.start_time + p.max_processing_time() for p in processes]))
    instance = get_or_instance(
        processes=processes, 
        scarcity=scarcity, 
        max_start_time=max_start_time,
        max_phases=MAX_PHASES,
        res_1_2_multiplier=RES_1_2_MULTIPLIER,
        res_1_3_multiplier=RES_1_3_MULTIPLIER,
    )
    instance_file = output_instance(
        instance, 
        scarcity, 
        max_start_time, 
        MAX_PHASES, 
        RES_1_2_MULTIPLIER, 
        RES_1_3_MULTIPLIER, 
        folder_name=db_instance_id
    )
    # Solver
    start_time = time.time()
    model, divisor = model_selector(model=solver, 
                                    n=instance["n"], 
                                    T=instance["T"], 
                                    M=instance["M"], 
                                    R=instance["R"], 
                                    E=instance["E"], 
                                    p=instance["p"], 
                                    L=instance["L"], 
                                    r=instance["r"], 
                                    O=instance["O"],
                                    ES=instance["ES"], 
                                    VP=instance["VP"], 
                                    obj=objective,
                                    timeout=timeout)
    model.optimize()
    solution_file = output_solution(model, 
                    solver,
                    scarcity, 
                    max_start_time, 
                    MAX_PHASES, 
                    RES_1_2_MULTIPLIER, 
                    RES_1_3_MULTIPLIER, 
                    folder_name=db_instance_id)
    is_feasible = not (
        model.status == GRB.INFEASIBLE or
        (model.SolCount == 0 and model.status == GRB.TIME_LIMIT)
    )
    if model.status == GRB.INFEASIBLE:
        print("\033[91mModel infeasible\033[0m")
    elif not is_feasible:
        print("\033[91mModel solution not found\033[0m")
    else: 
        print("\033[92mModel feasible\033[0m")
        print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
    db.add_solution(
        instance_id=db_instance_id, 
        solver=solver, 
        sol_file=solution_file, 
        instance_file=instance_file, 
        scarcity=scarcity, 
        divisor=divisor, 
        solved=True, 
        status=model.status,
        objective=objective,
        objective_val=model.objVal * divisor if is_feasible else None
    )
    pass


def main():
    db = Database("database.db")
    print("Generating instances...")
    instance_parameters = {
        "number_of_processes": 50, 
        "arrival_rate": 0.75,
        "batch_size": 2,
        "max_phases": 3,
        "min_base_duration": 2.0,
        "max_base_duration": 7.0,
        "min_resource_1_ratio": 1.0,
        "resource_1_ratio_center": 1.5,
        "resource_1_ratio_spread": 1.0,
        "min_resource_2_ratio": 1.0,
        "resource_2_ratio_center": 1.3,
        "resource_2_ratio_spread": 1.0,
        "res_1_2_multiplier": 2.0,
        "res_1_3_multiplier": 3.0,
        "job_3_multiplier": 1.0,
        "timeout": 600
    }
    processes = generate_instance(
        number_of_processes=instance_parameters["number_of_processes"], 
        arrival_rate=instance_parameters["arrival_rate"],
        batch_size=instance_parameters["batch_size"],
        max_phases=instance_parameters["max_phases"],
        min_base_duration=instance_parameters["min_base_duration"],
        max_base_duration=instance_parameters["max_base_duration"],
        min_resource_1_ratio=instance_parameters["min_resource_1_ratio"],
        resource_1_ratio_center=instance_parameters["resource_1_ratio_center"],
        resource_1_ratio_spread=instance_parameters["resource_1_ratio_spread"],
        min_resource_2_ratio=instance_parameters["min_resource_2_ratio"],
        resource_2_ratio_center=instance_parameters["resource_2_ratio_center"],
        resource_2_ratio_spread=instance_parameters["resource_2_ratio_spread"],
        res_1_2_multiplier=instance_parameters["res_1_2_multiplier"],
        res_1_3_multiplier=instance_parameters["res_1_3_multiplier"],
        job_3_multiplier=instance_parameters["job_3_multiplier"],
    )

    db_instance_id = db.add_instance(
        number_of_processes=instance_parameters["number_of_processes"], 
        arrival_rate=instance_parameters["arrival_rate"],
        batch_size=instance_parameters["batch_size"],
        max_phases=instance_parameters["max_phases"],
        min_base_duration=instance_parameters["min_base_duration"],
        max_base_duration=instance_parameters["max_base_duration"],
        min_resource_1_ratio=instance_parameters["min_resource_1_ratio"],
        resource_1_ratio_center=instance_parameters["resource_1_ratio_center"],
        resource_1_ratio_spread=instance_parameters["resource_1_ratio_spread"],
        min_resource_2_ratio=instance_parameters["min_resource_2_ratio"],
        resource_2_ratio_center=instance_parameters["resource_2_ratio_center"],
        resource_2_ratio_spread=instance_parameters["resource_2_ratio_spread"],
        res_1_2_multiplier=instance_parameters["res_1_2_multiplier"],
        res_1_3_multiplier=instance_parameters["res_1_3_multiplier"],
        job_3_multiplier=instance_parameters["job_3_multiplier"],
    )
    models = ["PDT", "PDDT", "SDT", "SDDT", "OODDT", "OOPDT", "OOPDDT", "MSEQCT"]
    scarcities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for model in models:
        for scarcity in scarcities:
            test_model(processes, solver=model, scarcity=scarcity, objective="flow-time", timeout=instance_parameters["timeout"], db=db, db_instance_id=db_instance_id)
    print(db.get_instances())
    print(db.get_solutions(instance_id=db_instance_id))
    create_tables(db_instance_id, db, calculate_unknown=True)


if __name__ == "__main__":
    main()



