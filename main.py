from instances.generator import generate_instance, get_min_max_demands, get_capacity
from instances.definitions import ResourceLevel, Mode, NetworkType
from instances.or_instance import get_or_instance
from gurobipy import GRB
import time

from src.pulse import pulse_model, pulse_model_disaggregated
from src.step import step_model, step_model_disaggregated
from src.onoff import onoff_model
from src.onoff_pulse import onoff_pulse_model, onoff_pulse_model_disaggregated
from src.vis_schedule import visualize_pulse_model


def model_selector(model: str, n, T, M, R, E, p, L, r, ES, VP):
    if model == "pulse" or model == "PDT":
        return pulse_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "pulse-disaggregated" or model == "PDDT":
        return pulse_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "step" or model == "SDT":
        return step_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "step-disaggregated" or model == "SDDT":
        return step_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "onoff" or model == "OODDT":
        return onoff_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "onoff-pulse" or model == "OOPDT":
        return onoff_pulse_model(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "onoff-pulse-disaggregated" or model == "OOPDDT":
        return onoff_pulse_model_disaggregated(n=n, T=T, M=M, R=R, E=E, p=p, L=L, r=r, ES=ES, VP=VP)
    if model == "continuous" or model == "MSEQCT":
        return continuous_model(n=n, T=T, M=M, R=R, E=E, ES=ES, VP=VP, p=p, L=L, r=r)
    raise ValueError(f"Unknown model {model}")


def main():
    print("Generating instances...")
    RES_1_2_MULTIPLIER = 2
    RES_1_3_MULTIPLIER = 3
    MAX_PHASES = 3
    processes = generate_instance(
        number_of_processes=20, 
        arrival_rate=0.7,
        max_phases=MAX_PHASES,
        min_base_duration=1.0,
        max_base_duration=5.0,
        min_resource_1_ratio=1.0,
        resource_1_ratio_center=1.4,
        resource_1_ratio_spread=.5,
        min_resource_2_ratio=1.0,
        resource_2_ratio_center=1.3,
        resource_2_ratio_spread=.5,
        res_1_2_multiplier=RES_1_2_MULTIPLIER,
        res_1_3_multiplier=RES_1_3_MULTIPLIER,
    )

    max_start_time = max([p.start_time + p.max_processing_time() for p in processes])

    scarcities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for scarcity in scarcities:
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
        model, divisor = model_selector(model="PDDT", n=instance["n"], T=instance["T"], M=instance["M"], R=instance["R"], E=instance["E"], p=instance["p"], L=instance["L"], r=instance["r"], ES=instance["ES"], VP=None)
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



