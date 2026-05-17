from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

from gurobipy import GRB
from instances.generator import generate_instance, get_min_max_demands
from instances.definitions import Process
from instances.or_instance import get_or_instance
from src.pulse import pulse_model, pulse_model_disaggregated
from src.step import step_model, step_model_disaggregated
from src.onoff import onoff_model
from src.onoff_pulse import onoff_pulse_model, onoff_pulse_model_disaggregated
from src.continuous import continuous_model
from src.vis_schedule import (
    visualize_pulse_model, visualize_continuous_model, visualize_onoff_model,
)
from database import Database

MODELS = ["PDT", "PDDT", "SDT", "SDDT", "OODDT", "OOPDT", "OOPDDT", "MSEQCT"]
SCARCITIES = [round(s * 0.1, 1) for s in range(11)]

DEFAULT_PARAMS: Dict[str, Any] = {
    "number_of_processes": 2,
    "arrival_rate": 0.5,
    "batch_size": 2,
    "max_phases": 3,
    "min_base_duration": 2.0,
    "max_base_duration": 5.0,
    "min_resource_ratio": 0.6,
    "resource_ratio_center": 1.5,
    "resource_ratio_spread": 1.0,
    "timeout": 600,
}

_MODEL_FNS = {
    "PDT": pulse_model, 
    "PDDT": pulse_model_disaggregated,
    "SDT": step_model,
    "SDDT": step_model_disaggregated,
    "OODDT": onoff_model,
    "OOPDT": onoff_pulse_model, 
    "OOPDDT": onoff_pulse_model_disaggregated,
    "MSEQCT": continuous_model,
}
# Explicit type overrides where default value type != desired argparse type
_PARAM_TYPES = {"batch_size": float}


def model_selector(model: str, **kwargs):
    if model not in _MODEL_FNS:
        raise ValueError(f"Unknown model: {model!r}")
    return _MODEL_FNS[model](**kwargs)


def test_model(
    processes: List[Process],
    n_resources: int,
    solver: str,
    scarcity: float,
    objective: str,
    min_max: List[Tuple[int, int]],
    max_phases: int = 3,
    timeout: int = 600,
    db: Optional[Database] = None,
    db_instance_id=None,
) -> None:
    if db is None:
        db = Database("database.db")
    if db_instance_id is None:
        raise ValueError("db_instance_id must be provided")

    print(f"Model: {solver}\tScarcity: {scarcity}")
    max_start_time = int(max(p.start_time + p.max_processing_time() for p in processes))
    instance = get_or_instance(
        processes=processes, 
        scarcity=scarcity, 
        max_start_time=max_start_time,
        n_resources=n_resources, 
        min_max=min_max, 
        max_phases=max_phases,
    )

    base = f"data/{db_instance_id}"
    os.makedirs(base, exist_ok=True)
    instance_file = f"{base}/instance_{scarcity}.json"
    with open(instance_file, "w") as f:
        json.dump(instance, f)

    t0 = time.time()
    model, divisor = model_selector(
        solver,
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
    model.update()
    print(
        f"Model: {model.NumVars} vars, "
        f"{model.NumConstrs} constraints, "
        f"{model.NumNZs} non-zeros"
    )
    model.optimize()

    sol_file = f"{base}/solution_{solver}_{scarcity}.json"
    model.write(sol_file)

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
        print(
            f"\033[1mRunning time: {time.time() - t0:.3f}s\t"
            f"Objective: {model.objVal * divisor:.0f}\033[0m"
        )
        vis_kw = dict(
            model=model, 
            n=instance["n"], 
            T=instance["T"], 
            M=instance["M"],
            R=instance["R"], 
            p=instance["p"], 
            r=instance["r"],
            processes=processes, 
            divisor=divisor,
        )
        if solver in {"PDT", "PDDT", "OOPDT", "OOPDDT"}:
            visualize_pulse_model(**vis_kw, filename=f"Schedule_{scarcity}")
        elif solver == "OODDT":
            visualize_onoff_model(**vis_kw, filename=f"Schedule_{scarcity}_onoff")
        elif solver == "MSEQCT":
            visualize_continuous_model(**vis_kw, filename=f"Schedule_{scarcity}_cont")

    db.add_solution(
        instance_id=db_instance_id, 
        solver=solver, 
        sol_file=sol_file,
        instance_file=instance_file, 
        scarcity=scarcity, 
        divisor=divisor,
        solved=True, 
        status=model.status, 
        objective=objective,
        objective_val=model.objVal * divisor if is_feasible else None,
    )


def create_tables(instance_id: int, db: Database) -> None:
    n_proc = db.get_instance(instance_id)["number_of_processes"]
    cols = "c|" * len(SCARCITIES)
    scarcity_row = " & ".join(f"{s:.1f}" for s in SCARCITIES)

    def _header(caption: str, label: str) -> str:
        return (
            "\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption} ({n_proc} processes). "
            "An entry with - indicates an infeasible instance.}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{|l|{cols}}}\n"
            f"\\hline\n\\(RS\\) & {scarcity_row} \\\\ \\hline\n"
        )

    makespan_table = _header("Makespan", "tab:makespan")
    runtime_table = _header("Runtime", "tab:runtime")
    footer = "\\end{tabular}\n\\end{table}\n"

    for model_name in MODELS:
        ms_row = rt_row = model_name
        for scarcity in SCARCITIES:
            sol = db.get_solution(instance_id, model_name, scarcity)
            if sol is None:
                ms_row += " & ?"; rt_row += " & ?"
                continue
            info = json.load(open(sol["sol_file"]))["SolutionInfo"]
            status, sol_count = info["Status"], info["SolCount"]
            if status == GRB.INFEASIBLE or (sol_count == 0 and status == GRB.TIME_LIMIT):
                ms_row += " & -"; rt_row += " & -"
            else:
                obj = int(info["ObjVal"])
                cell = f"\\textbf{{{obj}}}" if status == GRB.OPTIMAL else str(obj)
                ms_row += f" & {cell}"; rt_row += f" & {info['Runtime']:.2f}s"
        makespan_table += ms_row + " \\\\ \\hline\n"
        runtime_table += rt_row + " \\\\ \\hline\n"

    print(makespan_table + footer)
    print(runtime_table + footer)


def cmd_generate(args: argparse.Namespace) -> None:
    params = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    assert params["min_base_duration"] * params["min_resource_ratio"] >= 1.0, \
        "min_base_duration * min_resource_ratio must be >= 1.0"

    xml_files = (
        [e.split(",") if "," in e else e for e in args.xml_files]
        if getattr(args, "xml_files", None) is not None else None
    )
    gen_params = {k: v for k, v in params.items() if k != "timeout"}
    processes, global_resource_ids = generate_instance(xml_files=xml_files, **gen_params)
    n_resources = len(global_resource_ids)

    db = Database("database.db")
    db_instance_id = db.add_instance(**params, n_resources=n_resources, processes_file="")

    data_dir = f"data/{db_instance_id}"
    os.makedirs(data_dir, exist_ok=True)
    processes_file = f"{data_dir}/processes.pkl"
    with open(processes_file, "wb") as f:
        pickle.dump((processes, global_resource_ids), f)

    db.update_instance_processes_file(db_instance_id, processes_file)
    db.close()
    print(f"Instance ID : {db_instance_id}")
    print(f"Resources   : {n_resources}  {global_resource_ids}")
    print(f"Saved to    : {processes_file}")


def cmd_run(args: argparse.Namespace) -> None:
    db = Database("database.db")
    row = db.get_instance(args.instance_id)
    if row is None:
        print(f"Error: instance {args.instance_id} not found in database.")
        return

    with open(row["processes_file"], "rb") as f:
        processes, _ = pickle.load(f)

    min_max = get_min_max_demands(processes=processes, max_phases=row["max_phases"])
    for model_name in (args.models or MODELS):
        for scarcity in (args.scarcities or SCARCITIES):
            if db.get_solution(args.instance_id, model_name, scarcity) is not None:
                print(f"Skipping {model_name} @ {scarcity:.1f} (already in DB)")
                continue
            print("-" * 50)
            test_model(
                processes=processes, 
                n_resources=row["n_resources"],
                solver=model_name, 
                scarcity=scarcity, 
                objective=args.objective,
                min_max=min_max, 
                max_phases=row["max_phases"],
                timeout=row["timeout"], 
                db=db, 
                db_instance_id=args.instance_id,
            )

    create_tables(args.instance_id, db)
    db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resource-constrained scheduling experiment runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a new instance and save it to the DB")
    for k, v in DEFAULT_PARAMS.items():
        gen.add_argument(
            f"--{k.replace('_', '-')}",
            type=_PARAM_TYPES.get(k, type(v)),
            default=v,
        )
    gen.add_argument(
        "--xml-files", nargs="+", default=None, dest="xml_files",
        metavar="FILE[,FILE…]",
        help=(
            "RA-PST XML file(s) per phase. Each argument corresponds to one phase "
            "(cycled if fewer than --max-phases are given). Use a comma-separated "
            "list to define a per-process pool for that phase, e.g.: "
            "--xml-files phase0.xml \"phase1a.xml,phase1b.xml\""
        ),
    )
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser(
        "run", help="Solve models for an existing instance (skips already-solved pairs)"
    )
    run.add_argument("instance_id", type=int, help="DB instance ID returned by 'generate'")
    run.add_argument("--objective", default="flow-time", choices=["makespan", "flow-time"])
    run.add_argument(
        "--models", nargs="+", choices=MODELS, default=None, metavar="MODEL",
        help=f"Subset of models to run (default: all). Choices: {MODELS}",
    )
    run.add_argument(
        "--scarcities", nargs="+", type=float, default=None, metavar="S",
        help="Subset of scarcity levels to run (default: all 0.0–1.0)",
    )
    run.set_defaults(func=cmd_run)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
