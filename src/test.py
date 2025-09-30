import gurobipy as gp
from gurobipy import GRB
import json
import time
from continuous import continuous_model
from onoff import onoff_model
from step import step_model, step_model_disaggregated
from pulse import pulse_model, pulse_model_disaggregated
from onoff_pulse import onoff_pulse_model, onoff_pulse_model_disaggregated
import copy


def test_models(models, input_base, silent=True, time_limit=7200):
    results = []
    for model_name, model_init in models:
        print(f'Testing model `{model_name}`')
        input = copy.deepcopy(input_base)
        VP = [[i,j] for i in range(input["n"]) for j in range(input["n"]) if [i,j] not in input["E"] and [j,i] not in input["E"] and i != j]
        model, divisor = model_init(n=input["n"], T=input["T"], M=input["M"], R=input["R"], E=input["E"], VP=VP, p=input["p"], L=input["L"], r=input["r"])
        model.setParam('TimeLimit', time_limit)
        model.optimize()
        results.append({
            "model_name": model_name,
            "status": model.status,
            "objective": model.objVal * divisor if model.status != GRB.INFEASIBLE else -1,
            "gap": model.MIPGap,
            "runtime": model.Runtime,
            "variables": model.NumVars,
            "constraints": model.NumConstrs
        })
        if not silent: 
            if model.status == GRB.INFEASIBLE:
                print(f"\033[91mModel `{model_name}` infeasible\033[0m")
            else: 
                print(f"\033[92mModel `{model_name}` feasible\033[0m")
                print(f"Running time `{model_name}`: {model.Runtime} s")
    return results

def latex_table(stat):
    # Create latex table
    print(r"\begin{table}[!ht]")
    print(r"\centering")
    print(r"\begin{tabular}{l r r r r r}")
    print(r"\hline")
    print(r"Model & Objective & Gap & Runtime (s) & \#Vars & \#Cons \\")
    print(r"\hline")

    for stat in stats:
        print(f"{stat['model_name']} & "
              f"{stat['objective']:.3f} & {stat['gap']:.3f} & "
              f"{stat['runtime']:.3f} & {stat['variables']} & {stat['constraints']} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Solver statistics for different models (without status)}")
    print(r"\end{table}")

if __name__ == "__main__":
    # input_base = json.load(open("tests/simple.json"))
    input_base = json.load(open("tests/ra-pst-5.json"))
    stats = test_models([
        ("PDT", pulse_model),
        ("PDDT", pulse_model_disaggregated),
        ("SDT", step_model),
        ("SDDT", step_model_disaggregated),
        ("OODDT", onoff_model),
        ("OOPDT", onoff_pulse_model),
        ("OOPDDT", onoff_pulse_model_disaggregated),
        ("MSEQCT", continuous_model)
    ], input_base=input_base, silent=True, time_limit=300)
    # create table
    print(f"| {'model_name':<10} | {'status':<10} | {'objective':<10} | {'gap':<10} | {'runtime':<10} | {'# Vars':<10} | {'# Cons':<10} |")
    print(f"| {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} |")
    feasible = f"\033[92m{'FEASIBLE':<10}\033[0m"
    infeasible = f"\033[91m{'INFEASIBLE':<10}\033[0m"
    for stat in stats:
        print(f"| {stat['model_name']:<10} | {(infeasible if stat['status'] == GRB.INFEASIBLE else feasible):<10} | {stat['objective']:<10.3f} | {stat['gap']:<10.3f} | {stat['runtime']:8.3f} s | {stat['variables']:<10} | {stat['constraints']:<10} |")
    # Create latex table
    latex_table(stats)