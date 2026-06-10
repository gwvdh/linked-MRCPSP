from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any, Callable

from gurobipy import GRB

from database import Database
from instances.definitions import Process
from instances.generator import generate_instance, get_min_max_demands
from instances.or_instance import get_or_instance
from param import ARG_TYPES, DEFAULT_PARAMS, MODELS, SCARCITIES
from src.continuous import ContinuousModel
from src.onoff import OnoffModel
from src.onoff_pulse import OnoffPulseModel, OnoffPulseModelDisaggregated
from src.pulse import PulseModel, PulseModelDisaggregated
from src.step import StepModel, StepModelDisaggregated
from src.cp import constraint_programming_model
from src.vis_schedule import (
    visualize_continuous_model,
    visualize_onoff_model,
    visualize_pulse_model,
    visualize_cp_model,
)

MODEL_BUILDERS: dict[str, Callable[..., Any]] = {
    "PDT": PulseModel,
    "PDDT": PulseModelDisaggregated,
    "SDT": StepModel,
    "SDDT": StepModelDisaggregated,
    "OODDT": OnoffModel,
    "OOPDT": OnoffPulseModel,
    "OOPDDT": OnoffPulseModelDisaggregated,
    "MSEQCT": ContinuousModel,
}


def get_model_builder(model_name: str) -> Callable[..., Any]:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_BUILDERS[model_name]


def save_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_xml_files_arg(
    xml_files_arg: list[str] | None,
) -> list[str | list[str]] | None:
    """
    Parse CLI input of the form
        --xml-files phase0.xml "phase1a.xml,phase1b.xml"
    into
        ["phase0.xml", ["phase1a.xml", "phase1b.xml"]]
    """
    if xml_files_arg is None:
        return None
    return [entry.split(",") if "," in entry else entry for entry in xml_files_arg]


def solve_model_for_instance(
    *,
    processes: list[Process],
    n_resources: int,
    solver: str,
    scarcity: float,
    objective: str,
    min_max: list[tuple[int, int]],
    max_phases: int,
    timeout: int,
    db: Database,
    instance_id: int,
) -> None:
    """
    Build the OR instance, solve the selected model, save all artefacts,
    and record the result in the database.
    """
    print("-" * 60)
    print(f"Model = {solver}\tScarcity = {scarcity:.1f}")

    max_start_time = int(
        max(process.start_time + process.max_processing_time() for process in processes)
    )

    instance = get_or_instance(
        processes=processes,
        scarcity=scarcity,
        max_start_time=max_start_time,
        n_resources=n_resources,
        min_max=min_max,
        max_phases=max_phases,
    )

    data_dir = f"data/{instance_id}"
    os.makedirs(data_dir, exist_ok=True)

    instance_file = f"{data_dir}/instance_{scarcity:.1f}.json"
    with open(instance_file, "w") as f:
        json.dump(instance, f)

    scenario_id = db.upsert_scenario(
        instance_id=instance_id,
        scarcity=scarcity,
        instance_file=instance_file,
        T=instance["T"],
        n_jobs=instance["n"],
        n_resources=n_resources,
        capacities=instance["R"],
        descriptors={"objective": objective, "solver": solver},
    )

    builder = get_model_builder(solver)

    t0 = time.time()
    model = builder(
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
        LS=instance["LS"],
        VP=instance["VP"],
        obj=objective,
        timeout=timeout,
        processes=processes,
    )

    model.update()
    print(
        f"Model size: {model.number_of_variables()} vars, "
        f"{model.number_of_constraints()} constraints, "
        f"{model.number_of_nonzeros()} nonzeros"
    )

    model.solve()
    runtime = time.time() - t0

    sol_file = f"{data_dir}/solution_{solver}_{scarcity:.1f}.json"
    model.write(sol_file)

    if model.interrupted():
        print("\033[93mModel interrupted\033[0m")
        if model.sol_count() > 0:
            print(
                f"\033[92mFeasible solution found. \033[0m\033[1mRuntime = {runtime:.3f}s, "
                f"Objective = {model.get_objective():.0f}\033[0m"
            )
        else:
            print("\033[91mNo incumbent solution available.")
    elif not model.is_feasible():
        print("\033[91mModel infeasible.\033[0m")
    elif model.is_timed_out():
        print("\033[91mNo solution found within the time limit.\033[0m")
    else:
        print(
            f"\033[92mFeasible solution found. \033[0m\033[1mRuntime = {runtime:.3f}s, "
            f"Objective = {model.get_objective():.0f}\033[0m"
        )
        model.visualize(f"Schedule_{solver}_{scarcity:.1f}")

    db.record_solution(
        instance_id=instance_id,
        scenario_id=scenario_id,
        solver=solver,
        sol_file=sol_file,
        instance_file=instance_file,
        scarcity=scarcity,
        divisor=model.divisor,
        objective=objective,
        status=model.status(),
        finished=not model.interrupted(),
        feasible=model.is_feasible(),
        optimal=model.is_optimal(),
        objective_val=model.get_objective() if model.is_feasible() else None,
        lower_bound=model.lower_bound(),
        cpm_lb=model.cpm_lb(),
        solver_time=model.solver_time(),
        runtime=runtime,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    params = {key: getattr(args, key) for key in DEFAULT_PARAMS}

    if params["min_base_duration"] * params["min_resource_ratio"] < 1.0:
        raise ValueError(
            "min_base_duration * min_resource_ratio must be at least 1.0"
        )

    xml_files = parse_xml_files_arg(args.xml_files)

    generator_params = {k: v for k, v in params.items() if k != "timeout"}
    processes, global_resource_ids = generate_instance(
        xml_files=xml_files,
        **generator_params,
    )

    n_resources = len(global_resource_ids)

    with Database(args.database) as db:
        instance_id = db.add_instance(
            **params,
            n_resources=n_resources,
            processes_file="",
            xml_files=xml_files,
            global_resource_ids=global_resource_ids,
            generator_version="compact-rewrite",
            generation_metadata={},
        )

        data_dir = f"data/{instance_id}"
        os.makedirs(data_dir, exist_ok=True)

        processes_file = f"{data_dir}/processes.pkl"
        save_pickle(processes_file, (processes, global_resource_ids))
        db.update_instance_processes_file(instance_id, processes_file)

    print(f"Instance ID : {instance_id}")
    print(f"Resources   : {n_resources} {global_resource_ids}")
    print(f"Saved to    : {processes_file}")


def cmd_run(args: argparse.Namespace) -> None:
    with Database(args.database) as db:
        row = db.get_instance(args.instance_id)
        if row is None:
            print(f"Error: instance {args.instance_id} not found.")
            return

        processes, _ = load_pickle(row["processes_file"])
        models = args.models or MODELS
        scarcities = args.scarcities or SCARCITIES

        min_max = get_min_max_demands(
            processes=processes,
            max_phases=row["max_phases"],
        )

        for model_name in models:
            for scarcity in scarcities:
                existing = db.get_solution(args.instance_id, model_name, scarcity)
                if existing is not None and existing["solved"]:
                    print(f"Skipping {model_name} @ {scarcity:.1f} (already solved)")
                    continue

                solve_model_for_instance(
                    processes=processes,
                    n_resources=row["n_resources"],
                    solver=model_name,
                    scarcity=scarcity,
                    objective=args.objective,
                    min_max=min_max,
                    max_phases=row["max_phases"],
                    timeout=row["timeout"],
                    db=db,
                    instance_id=args.instance_id,
                )

        objective_table, runtime_table = db.make_latex_tables(
            args.instance_id,
            models,
            scarcities,
        )
        print(objective_table)
        print(runtime_table)


def cmd_run_all(args: argparse.Namespace) -> None:
    with Database(args.database) as db:
        instance_ids = [row["id"] for row in db.get_instances()]

    for instance_id in instance_ids:
        args.instance_id = instance_id
        args.models = None
        args.scarcities = None
        print(f"Running instance {instance_id}")
        cmd_run(args)


def cmd_run_all_in_dataset(args: argparse.Namespace) -> None:
    with Database(args.database) as db:
        pending = db.get_unsolved_solutions_for_dataset(args.dataset_id)

        print(f"Solving {len(pending)} unsolved model-instance pairs")

        for sol in pending:
            instance_id = sol["instance_id"]
            row = db.get_instance(instance_id)
            if row is None:
                continue

            processes, _ = load_pickle(row["processes_file"])
            min_max = get_min_max_demands(
                processes=processes,
                max_phases=row["max_phases"],
            )

            solve_model_for_instance(
                processes=processes,
                n_resources=row["n_resources"],
                solver=sol["solver"],
                scarcity=sol["scarcity"],
                objective=sol["objective"],
                min_max=min_max,
                max_phases=row["max_phases"],
                timeout=row["timeout"],
                db=db,
                instance_id=instance_id,
            )


def cmd_create_dataset(args: argparse.Namespace) -> None:
    params = {key: getattr(args, key) for key in DEFAULT_PARAMS}
    xml_files = parse_xml_files_arg(args.xml_files)
    scarcities = args.scarcities or SCARCITIES

    with Database(args.database) as db:
        dataset_id = db.add_dataset(
            name=args.name,
            description=args.description,
            n_processes=args.number_of_processes,
            n_instances=args.n_instances,
        )

        for _ in range(args.n_instances):
            generator_params = {k: v for k, v in params.items() if k != "timeout"}
            processes, global_resource_ids = generate_instance(
                xml_files=xml_files,
                **generator_params,
            )

            n_resources = len(global_resource_ids)

            instance_id = db.add_instance(
                **params,
                n_resources=n_resources,
                processes_file="",
                xml_files=xml_files,
                global_resource_ids=global_resource_ids,
                generator_version="compact-rewrite",
                generation_metadata={},
            )

            data_dir = f"data/{instance_id}"
            os.makedirs(data_dir, exist_ok=True)

            processes_file = f"{data_dir}/processes.pkl"
            save_pickle(processes_file, (processes, global_resource_ids))
            db.update_instance_processes_file(instance_id, processes_file)

            db.add_instance_to_dataset(instance_id, dataset_id)

            for scarcity in scarcities:
                for model_name in MODELS:
                    db.create_pending_solution(
                        instance_id=instance_id,
                        solver=model_name,
                        scarcity=scarcity,
                        objective=args.objective,
                    )

            print(f"Created instance {instance_id}")

        print(f"Created dataset {dataset_id}")


def cmd_table(args: argparse.Namespace) -> None:
    with Database(args.database) as db:
        models = args.models or MODELS
        scarcities = args.scarcities or SCARCITIES

        if args.instance_id is not None:
            objective_table, runtime_table = db.make_latex_tables(
                args.instance_id,
                models,
                scarcities,
            )
        else:
            objective_table, runtime_table = db.make_dataset_latex_tables(
                args.dataset_id,
                models,
                scarcities,
            )

        print(objective_table)
        print(runtime_table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resource-constrained scheduling experiment runner"
    )
    parser.add_argument(
        "--database",
        default="database.db",
        help="SQLite database path",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a single instance")
    for key, default in DEFAULT_PARAMS.items():
        gen.add_argument(
            f"--{key.replace('_', '-')}",
            type=ARG_TYPES.get(key, type(default)),
            default=default,
        )
    gen.add_argument(
        "--xml-files",
        nargs="+",
        default=None,
        dest="xml_files",
        metavar="FILE[,FILE...]",
        help=(
            "RA-PST XML file(s) per phase. Each argument corresponds to one "
            "phase (cycled if fewer than --max-phases are given). Use a "
            "comma-separated list to define a per-phase pool, e.g. "
            '--xml-files phase0.xml "phase1a.xml,phase1b.xml"'
        ),
    )
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser(
        "run",
        help="Solve models for an existing instance",
    )
    run.add_argument(
        "instance_id",
        type=int,
        help="Instance ID in the database",
    )
    run.add_argument(
        "--objective",
        default="flow-time",
        choices=["makespan", "flow-time"],
    )
    run.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=None,
        metavar="MODEL",
        help=f"Subset of models to run (default: all). Choices: {MODELS}",
    )
    run.add_argument(
        "--scarcities",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
        help="Subset of scarcity levels to run (default: all 0.2-1.0)",
    )
    run.set_defaults(func=cmd_run)

    run_all = sub.add_parser(
        "run-all",
        help="Solve all instances currently stored in the database",
    )
    run_all.add_argument(
        "--objective",
        default="flow-time",
        choices=["makespan", "flow-time"],
    )
    run_all.set_defaults(func=cmd_run_all)

    run_dataset = sub.add_parser(
        "run-all-in-dataset",
        help="Solve all unsolved model-instance pairs in a dataset",
    )
    run_dataset.add_argument(
        "dataset_id",
        type=int,
        help="Dataset ID in the database",
    )
    run_dataset.set_defaults(func=cmd_run_all_in_dataset)

    create_dataset = sub.add_parser(
        "create-dataset",
        help="Create a dataset of multiple generated instances",
    )
    create_dataset.add_argument("name", type=str, help="Dataset name")
    create_dataset.add_argument(
        "--description",
        type=str,
        default="",
        help="Dataset description",
    )
    for key, default in DEFAULT_PARAMS.items():
        create_dataset.add_argument(
            f"--{key.replace('_', '-')}",
            type=ARG_TYPES.get(key, type(default)),
            default=default,
        )
    create_dataset.add_argument(
        "--n-instances",
        dest="n_instances",
        type=int,
        default=10,
        help="Number of instances to generate",
    )
    create_dataset.add_argument(
        "--scarcities",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
        help="Scarcity levels for which pending solutions are created",
    )
    create_dataset.add_argument(
        "--objective",
        default="flow-time",
        choices=["makespan", "flow-time"],
    )
    create_dataset.add_argument(
        "--xml-files",
        nargs="+",
        default=None,
        dest="xml_files",
        metavar="FILE[,FILE...]",
        help=(
            "RA-PST XML file(s) per phase. Each argument corresponds to one "
            "phase (cycled if fewer than --max-phases are given). Use a "
            "comma-separated list to define a per-phase pool, e.g. "
            '--xml-files phase0.xml "phase1a.xml,phase1b.xml"'
        ),
    )
    create_dataset.set_defaults(func=cmd_create_dataset)

    table = sub.add_parser(
        "table",
        help="Print LaTeX tables for one instance or one dataset",
    )
    table_group = table.add_mutually_exclusive_group(required=True)
    table_group.add_argument(
        "--instance-id",
        type=int,
        default=None,
        dest="instance_id",
        help="Instance ID in the database",
    )
    table_group.add_argument(
        "--dataset-id",
        type=int,
        default=None,
        dest="dataset_id",
        help="Dataset ID in the database",
    )
    table.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=None,
        metavar="MODEL",
    )
    table.add_argument(
        "--scarcities",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
    )
    table.set_defaults(func=cmd_table)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
