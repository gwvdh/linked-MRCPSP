from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from gurobipy import GRB

from instances.generator import generate_instance, get_min_max_demands, get_capacity
from instances.definitions import Process
from instances.or_instance import get_or_instance
from instances.xml_parser import RA_PST

from src.pulse import pulse_model, pulse_model_disaggregated
from src.step import step_model, step_model_disaggregated
from src.onoff import onoff_model
from src.onoff_pulse import onoff_pulse_model, onoff_pulse_model_disaggregated
from src.continuous import continuous_model
from src.vis_schedule import visualize_pulse_model, visualize_continuous_model
from database import Database

XML_FILE = "rapst/full_rapst_permit.xml"

MODELS = ["PDT", "PDDT", "SDT", "SDDT", "OODDT", "OOPDT", "OOPDDT", "MSEQCT"]
SCARCITIES = [round(s * 0.1, 1) for s in range(11)]
#MODELS = ["PDT", "MSEQCT"]
#SCARCITIES = [0.0]


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


def model_selector(
    model: str,
    n: int,
    T: int,
    M: int,
    R: List[int],
    E: List[List[int]],
    p: List[List[int]],
    L: List[List[int]],
    r: List[List[List[int]]],
    O: List[int],
    ES: List[int],
    VP: List[List[int]],
    obj: str = "makespan",
    timeout: int = 600,
):
    """Solve the given model with the given parameters."""
    kwargs = dict(
        n=n, T=T, M=M, R=R, E=E, p=p, L=L,
        r=r, O=O, ES=ES, VP=VP, obj=obj, timeout=timeout,
    )
    dispatch = {
        "PDT":    pulse_model,
        "PDDT":   pulse_model_disaggregated,
        "SDT":    step_model,
        "SDDT":   step_model_disaggregated,
        "OODDT":  onoff_model,
        "OOPDT":  onoff_pulse_model,
        "OOPDDT": onoff_pulse_model_disaggregated,
        "MSEQCT": continuous_model,
    }
    if model not in dispatch:
        raise ValueError(f"Unknown model: {model!r}")
    return dispatch[model](**kwargs)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def output_instance(
    instance: Dict[str, Any],
    scarcity: float,
    folder_name: str = "",
) -> str:
    """Write the given instance to a file."""
    _ensure_dir(f"data/{folder_name}")
    filename = f"data/{folder_name}/instance_{scarcity}.json"
    with open(filename, "w") as f:
        json.dump(instance, f)
    return filename


def output_solution(
    model,
    model_name: str,
    scarcity: float,
    folder_name: str = "",
) -> str:
    """Write the given gurobi solution to a file."""
    _ensure_dir(f"data/{folder_name}")
    sol_file = f"data/{folder_name}/solution_{model_name}_{scarcity}.json"
    model.write(sol_file)
    return sol_file


# ---------------------------------------------------------------------------
# Single model run
# ---------------------------------------------------------------------------


def test_model(
    processes: List[Process],
    ra_pst: RA_PST,
    solver: str,
    scarcity: float,
    objective: str,
    max_phases: int = 3,
    timeout: int = 600,
    db: Optional[Database] = None,
    db_instance_id=None,
) -> None:
    """
    Solve the given set of processes using the given solver with the given parameters.
    :param processes: The processes to solve.
    :param ra_pst: The RA-PST object.
    :param solver: The solver to use.
    :param scarcity: The scarcity of the instance.
    :param objective: The objective to optimize.
    :param max_phases: The maximum number of phases.
    :param timeout: The maximum runtime of the solver.
    :param db: The SQLITE database object.
    :param db_instance_id: The ID of the instance to solve.
    """
    if db is None:
        db = Database("database.db")
    if db_instance_id is None:
        raise ValueError("db_instance_id must be provided")

    print(f"Model: {solver}\tScarcity: {scarcity}")

    max_start_time = int(
        max(p.start_time + p.max_processing_time() for p in processes)
    )

    instance = get_or_instance(
        processes=processes,
        scarcity=scarcity,
        max_start_time=int(max_start_time/2),
        ra_pst=ra_pst,
        max_phases=max_phases,
    )
    instance_file = output_instance(
        instance, scarcity, folder_name=str(db_instance_id)
    )

    t0 = time.time()
    model, divisor = model_selector(
        model=solver,
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
        timeout=timeout,
    )
    model.optimize()

    solution_file = output_solution(
        model, solver, scarcity, folder_name=str(db_instance_id)
    )

    is_feasible = not (
        model.status == GRB.INFEASIBLE
        or (model.SolCount == 0 and model.status == GRB.TIME_LIMIT)
    )

    if model.status == GRB.INFEASIBLE:
        print("\033[91mModel infeasible\033[0m")
    elif not is_feasible:
        print("\033[91mNo solution found within time limit\033[0m")
    else:
        print("\033[92mFeasible solution found\033[0m")
        print(f"\033[1mRunning time: {time.time() - t0:.3f}s\tObjective: {model.objVal * divisor:.1f}\033[0m")
        if solver in ("PDT", "PDDT", "OOPDT", "OOPDDT"):
            visualize_pulse_model(
                model=model,
                n=instance["n"],
                T=instance["T"],
                M=instance["M"],
                R=instance["R"],
                p=instance["p"],
                r=instance["r"],
                processes=processes,
                divisor=divisor,
                filename=f"Schedule_{scarcity}",
            )
        elif solver in ("MSEQCT"):
            visualize_continuous_model(
                model=model,
                n=instance["n"],
                T=instance["T"],
                M=instance["M"],
                R=instance["R"],
                p=instance["p"],
                r=instance["r"],
                processes=processes,
                divisor=divisor,
                filename=f"Schedule_{scarcity}_cont",
            )

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
        objective_val=model.objVal * divisor if is_feasible else None,
    )


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def create_tables(instance_id, db: Database) -> None:
    """Create a LaTeX table for the given instance and its solutions."""
    instance_row = db.get_instance(instance_id)
    n_processes = instance_row[1]

    def _header(caption: str, label: str) -> str:
        cols = "c|" * len(SCARCITIES)
        scarcity_row = " & ".join(f"{s:.1f}" for s in SCARCITIES)
        return (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\caption{{{caption} ({n_processes} processes). "
            "An entry with - indicates an infeasible instance.}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{|l|{cols}}}\n"
            "\\hline\n"
            f"\\(RS\\) & {scarcity_row} \\\\ \\hline\n"
        )

    makespan_table = _header("Makespan", "tab:makespan")
    runtime_table = _header("Runtime", "tab:runtime")

    for model_name in MODELS:
        makespan_table += model_name
        runtime_table += model_name

        for scarcity in SCARCITIES:
            solution = db.get_solution(
                instance_id=instance_id,
                model_name=model_name,
                scarcity=scarcity,
            )
            if solution is None:
                makespan_table += " & ?"
                runtime_table += " & ?"
                continue

            sol_data = json.load(open(solution[3]))
            status = sol_data["SolutionInfo"]["Status"]
            sol_count = sol_data["SolutionInfo"]["SolCount"]

            if status == GRB.INFEASIBLE or (
                sol_count == 0 and status == GRB.TIME_LIMIT
            ):
                makespan_table += " & -"
                runtime_table += " & -"
            else:
                obj_val = sol_data["SolutionInfo"]["ObjVal"]
                runtime = sol_data["SolutionInfo"]["Runtime"]
                cell = (
                    f"\\textbf{{{obj_val:.1f}}}"
                    if status == GRB.OPTIMAL
                    else f"{obj_val:.1f}"
                )
                makespan_table += f" & {cell}"
                runtime_table += f" & {runtime:.2f}s"

        makespan_table += " \\\\ \\hline\n"
        runtime_table += " \\\\ \\hline\n"

    footer = "\\end{tabular}\n\\end{table}\n"
    print(makespan_table + footer)
    print(runtime_table + footer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    db = Database("database.db")
    ra_pst = RA_PST(XML_FILE)

    instance_parameters = {
        "number_of_processes": 10,
        "arrival_rate": 0.75,
        "batch_size": 2,
        "max_phases": 3,
        "min_base_duration": 2.0,
        "max_base_duration": 5.0,
        "min_resource_ratio": 1.0,
        "resource_ratio_center": 1.5,
        "resource_ratio_spread": 1.0,
        "timeout": 600,
    }

    processes = generate_instance(
        number_of_processes=instance_parameters["number_of_processes"],
        arrival_rate=instance_parameters["arrival_rate"],
        batch_size=instance_parameters["batch_size"],
        max_phases=instance_parameters["max_phases"],
        min_base_duration=instance_parameters["min_base_duration"],
        max_base_duration=instance_parameters["max_base_duration"],
        min_resource_ratio=instance_parameters["min_resource_ratio"],
        resource_ratio_center=instance_parameters["resource_ratio_center"],
        resource_ratio_spread=instance_parameters["resource_ratio_spread"],
    )

    db_instance_id = db.add_instance(**instance_parameters)

    for model in MODELS:
        for scarcity in SCARCITIES:
            test_model(
                processes=processes,
                ra_pst=ra_pst,
                solver=model,
                scarcity=scarcity,
                objective="flow-time",
                max_phases=instance_parameters["max_phases"],
                timeout=instance_parameters["timeout"],
                db=db,
                db_instance_id=db_instance_id,
            )

    print(db.get_instances())
    print(db.get_solutions(instance_id=db_instance_id))
    create_tables(db_instance_id, db)


if __name__ == "__main__":
    main()
