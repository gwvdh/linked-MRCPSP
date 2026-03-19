from instances.generator import generate_instance, get_min_max_demands, get_capacity
from instances.definitions import ResourceLevel, Mode, NetworkType
from instances.or_instance import get_or_instance
from gurobipy import GRB
import time
import copy

from src.pulse import pulse_model, pulse_model_disaggregated
from src.step import step_model, step_model_disaggregated
from src.onoff import onoff_model
from src.onoff_pulse import onoff_pulse_model, onoff_pulse_model_disaggregated
from src.continuous import continuous_model
from src.vis_schedule import visualize_pulse_model


def model_selector(model: str, n, T, M, R, E, p, L, r, O, ES, VP, obj="makespan"):
    if model == "pulse" or model == "PDT":
        return pulse_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "pulse-disaggregated" or model == "PDDT":
        return pulse_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "step" or model == "SDT":
        return step_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "step-disaggregated" or model == "SDDT":
        return step_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "onoff" or model == "OODDT":
        return onoff_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "onoff-pulse" or model == "OOPDT":
        return onoff_pulse_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "onoff-pulse-disaggregated" or model == "OOPDDT":
        return onoff_pulse_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, O=O, ES=ES, VP=VP, obj=obj)
    if model == "continuous" or model == "MSEQCT":
        return continuous_model(n=n, T=T, M=M, R=R, E=E, ES=ES, VP=VP, p=p, L=L, r=r, O=O, obj=obj)
    raise ValueError(f"Unknown model {model}")


def test_all_models(processes, MAX_PHASES=3, RES_1_2_MULTIPLIER=2.0, RES_1_3_MULTIPLIER=3.0):
    models = ["PDT", "PDDT", "SDT", "SDDT", "OODDT", "OOPDT", "OOPDDT", "MSEQCT"]
    scarcities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    max_start_time = int(max([p.start_time + p.max_processing_time() for p in processes]))
    #max_start_time = 60
    print(f'Max start time: {max_start_time}')
    latex_table = ""
    latex_table += "\\begin{table}[ht]\n"
    latex_table += "\\centering\n"
    latex_time_table = copy.deepcopy(latex_table)
    latex_table += "\\caption{Makespan (" + f"{len(processes)}" + " processes). An entry with - indicates an infeasible instance.}\n"
    latex_table += "\\label{tab:makespan}\n"
    latex_table += "\\begin{tabular}{|l|" + "c|" * len(scarcities) + "}\n"
    latex_table += "\\hline\n"
    latex_table += "\(RS\) &" + " & ".join([f"{scarcity:.1f}" for scarcity in scarcities]) + " \\\\ \\hline\n"
    latex_time_table += "\\caption{Runtime (" + f"{len(processes)}" + " processes). An entry with - indicates an infeasible instance.}\n"
    latex_time_table += "\\label{tab:runtime}\n"
    latex_time_table += "\\begin{tabular}{|l|" + "c|" * len(scarcities) + "}\n"
    latex_time_table += "\\hline\n"
    latex_time_table += "\(RS\) &" + " & ".join([f"{scarcity:.1f}" for scarcity in scarcities]) + " \\\\ \\hline\n"
    for model_name in models:
        latex_table += f"{model_name}"
        latex_time_table += f"{model_name}"
        for scarcity in scarcities:
            print(f"Model: {model_name}\tScarcity: {scarcity}")
            instance = get_or_instance(
                processes=processes, 
                scarcity=scarcity, 
                max_start_time=max_start_time,
                max_phases=MAX_PHASES,
                res_1_2_multiplier=RES_1_2_MULTIPLIER,
                res_1_3_multiplier=RES_1_3_MULTIPLIER,
            )
            # Solver
            start_time = time.time()
            model, divisor = model_selector(model=model_name, 
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
                                            obj="flow-time")
            model.optimize()
            if model.status == GRB.INFEASIBLE:
                print("\033[91mModel infeasible\033[0m")
                latex_table += " & - "
                latex_time_table += " & - "
            else: 
                print("\033[92mModel feasible\033[0m")
                print(f"\033[1mRunning time: {time.time() - start_time:.3f} s\033[0m")
                if model_name == "PDT" or model_name == "PDDT" or model_name == "OOPDT" or model_name == "OOPDDT":
                    visualize_pulse_model(model=model, 
                                        n=instance["n"], 
                                        T=instance["T"], 
                                        M=instance["M"], 
                                        R=instance["R"], 
                                        p=instance["p"], 
                                        r=instance["r"], 
                                        processes=processes,
                                        divisor=divisor, 
                                        filename=f"Schedule_{scarcity}"
                                        )
                elif model_name == "SDT" or model_name == "SDDT":
                    pass
                elif model_name == "OODDT":
                    pass
                elif model_name == "MSEQCT":
                    pass
                for var in model.getVars():
                    print(f"{var.varName}: {var.x}") if var.x > 0 else None
                print(f"Objective: {model.objVal * divisor}")
                latex_table += " & " + f"{model.objVal * divisor:.1f}"
                latex_time_table += f" & {model.Runtime:.2f}s"
        latex_table += " \\\\ \\hline\n"
        latex_time_table += " \\\\ \\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    latex_time_table += "\\end{tabular}\n"
    latex_time_table += "\\end{table}"
    print(latex_table)
    print(latex_time_table)


def main():
    print("Generating instances...")
    RES_1_2_MULTIPLIER = 2.0
    RES_1_3_MULTIPLIER = 3.0
    JOB_3_MULTIPLIER = 1.0
    MAX_PHASES = 3
    processes = generate_instance(
        number_of_processes=10, 
        arrival_rate=3,
        max_phases=MAX_PHASES,
        min_base_duration=3.0,
        max_base_duration=8.0,
        min_resource_1_ratio=1.0,
        resource_1_ratio_center=1.5,
        resource_1_ratio_spread=1.0,
        min_resource_2_ratio=1.0,
        resource_2_ratio_center=1.3,
        resource_2_ratio_spread=1.0,
        res_1_2_multiplier=RES_1_2_MULTIPLIER,
        res_1_3_multiplier=RES_1_3_MULTIPLIER,
        job_3_multiplier=JOB_3_MULTIPLIER,
    )

    test_all_models(processes)


if __name__ == "__main__":
    main()



