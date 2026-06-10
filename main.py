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
from src.cp import ConstraintProgrammingModel
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
    "CP": ConstraintProgrammingModel,
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
        var_count=model.number_of_variables(),
        const_count=model.number_of_constraints(),
    )

def cmd_generate(args: argparse.Namespace) -> None:
    """
    Generate exactly one instance with the provided generator arguments.

    By default this only creates the instance row and stores the generated
    processes. If --create-pending is provided, pending solution rows are also
    created for the selected models/scarcities.
    """
    params = {key: getattr(args, key) for key in DEFAULT_PARAMS}
    validate_generator_params(params)

    xml_files = parse_xml_files_arg(args.xml_files)
    generator_params = {key: value for key, value in params.items() if key != "timeout"}

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

        if args.create_pending:
            models = args.models or MODELS
            scarcities = args.scarcities or SCARCITIES

            for scarcity in scarcities:
                for model_name in models:
                    db.create_pending_solution(
                        instance_id=instance_id,
                        solver=model_name,
                        scarcity=scarcity,
                        objective=args.objective,
                    )

    print(f"Instance ID : {instance_id}")
    print(f"Resources   : {n_resources} {global_resource_ids}")
    print(f"Saved to    : {processes_file}")

    if args.create_pending:
        print("Pending solution rows were created.")


def validate_generator_params(params: dict[str, Any]) -> None:
    if params["min_base_duration"] * params["min_resource_ratio"] < 1.0:
        raise ValueError(
            "min_base_duration * min_resource_ratio must be at least 1.0"
        )


def get_solution_rows_for_instance(
    db: Database,
    instance_id: int,
    force: bool,
) -> list[Any]:
    if force:
        db.cur.execute(
            """
            SELECT *
            FROM solution
            WHERE instance_id = ?
            ORDER BY scarcity, solver
            """,
            (instance_id,),
        )
    else:
        db.cur.execute(
            """
            SELECT *
            FROM solution
            WHERE instance_id = ?
              AND COALESCE(solved, 0) = 0
            ORDER BY scarcity, solver
            """,
            (instance_id,),
        )

    return db.cur.fetchall()


def get_all_solution_rows(db: Database, force: bool) -> list[Any]:
    if force:
        db.cur.execute(
            """
            SELECT *
            FROM solution
            ORDER BY instance_id, scarcity, solver
            """
        )
    else:
        db.cur.execute(
            """
            SELECT *
            FROM solution
            WHERE COALESCE(solved, 0) = 0
            ORDER BY instance_id, scarcity, solver
            """
        )

    return db.cur.fetchall()


def get_solution_rows_for_dataset(
    db: Database,
    dataset_id: int,
    force: bool,
) -> list[Any]:
    db.cur.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    dataset = db.cur.fetchone()
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    if force:
        db.cur.execute(
            """
            SELECT solution.*
            FROM solution
            INNER JOIN instance_dataset
                ON solution.instance_id = instance_dataset.instance_id
            WHERE instance_dataset.dataset_id = ?
            ORDER BY solution.instance_id, solution.scarcity, solution.solver
            """,
            (dataset_id,),
        )
    else:
        db.cur.execute(
            """
            SELECT solution.*
            FROM solution
            INNER JOIN instance_dataset
                ON solution.instance_id = instance_dataset.instance_id
            WHERE instance_dataset.dataset_id = ?
              AND COALESCE(solution.solved, 0) = 0
            ORDER BY solution.instance_id, solution.scarcity, solution.solver
            """,
            (dataset_id,),
        )

    return db.cur.fetchall()


def solve_solution_rows(
    *,
    db: Database,
    rows: list[Any],
    objective_fallback: str,
) -> None:
    """
    Solve the given rows from the solution table.

    A row determines:
    - instance_id
    - solver/model
    - scarcity
    - objective, with objective_fallback used for legacy rows where objective
      is NULL.
    """
    if not rows:
        print("No solution rows to solve.")
        return

    instance_cache: dict[int, tuple[Any, list[Process], list[tuple[int, int]]]] = {}

    print(f"Solving {len(rows)} solution row(s).")

    for sol in rows:
        instance_id = int(sol["instance_id"])
        solver = sol["solver"]
        scarcity = sol["scarcity"]

        if solver not in MODELS:
            print(
                f"Skipping solution row {sol['id']}: unknown solver/model {solver!r}"
            )
            continue

        if scarcity is None:
            print(f"Skipping solution row {sol['id']}: scarcity is NULL")
            continue

        if instance_id not in instance_cache:
            instance = db.get_instance(instance_id)
            if instance is None:
                print(f"Skipping solution row {sol['id']}: instance not found")
                continue

            if not instance["processes_file"]:
                print(
                    f"Skipping solution row {sol['id']}: "
                    f"instance {instance_id} has no processes_file"
                )
                continue

            processes, _ = load_pickle(instance["processes_file"])
            min_max = get_min_max_demands(
                processes=processes,
                max_phases=instance["max_phases"],
            )

            instance_cache[instance_id] = (instance, processes, min_max)

        instance, processes, min_max = instance_cache[instance_id]

        objective = sol["objective"] or objective_fallback

        solve_model_for_instance(
            processes=processes,
            n_resources=instance["n_resources"],
            solver=solver,
            scarcity=float(scarcity),
            objective=objective,
            min_max=min_max,
            max_phases=instance["max_phases"],
            timeout=instance["timeout"],
            db=db,
            instance_id=instance_id,
        )


def cmd_run(args: argparse.Namespace) -> None:
    """
    Solve all unsolved solution-table entries for a single instance.

    With --force, all existing solution-table entries for the instance are
    solved again, including entries already marked as solved.
    """
    with Database(args.database) as db:
        instance = db.get_instance(args.instance_id)
        if instance is None:
            print(f"Error: instance {args.instance_id} not found.")
            return

        rows = get_solution_rows_for_instance(
            db=db,
            instance_id=args.instance_id,
            force=args.force,
        )

        solve_solution_rows(
            db=db,
            rows=rows,
            objective_fallback=args.objective,
        )


def cmd_run_all(args: argparse.Namespace) -> None:
    """
    Solve all unsolved solution-table entries in the database.

    With --force, all existing solution-table entries are solved again.
    """
    with Database(args.database) as db:
        rows = get_all_solution_rows(db=db, force=args.force)

        solve_solution_rows(
            db=db,
            rows=rows,
            objective_fallback=args.objective,
        )


def cmd_run_dataset(args: argparse.Namespace) -> None:
    """
    Solve all unsolved solution-table entries belonging to a dataset.

    With --force, all existing solution-table entries belonging to the dataset
    are solved again.
    """
    with Database(args.database) as db:
        rows = get_solution_rows_for_dataset(
            db=db,
            dataset_id=args.dataset_id,
            force=args.force,
        )

        solve_solution_rows(
            db=db,
            rows=rows,
            objective_fallback=args.objective,
        )


def cmd_create_dataset(args: argparse.Namespace) -> None:
    """
    Create a dataset of generated instances.

    Important behavior:
    - args.n_instances means "instances per scarcity".
    - Each generated instance is used for exactly one scarcity.
    - When a new scarcity is considered, new instances are generated.
    - Instances are not reused across different scarcity levels.
    """
    params = {key: getattr(args, key) for key in DEFAULT_PARAMS}
    validate_generator_params(params)

    xml_files = parse_xml_files_arg(args.xml_files)
    scarcities = args.scarcities or SCARCITIES
    models = args.models or MODELS

    total_instances = args.n_instances * len(scarcities)

    with Database(args.database) as db:
        dataset_id = db.add_dataset(
            name=args.name,
            description=args.description,
            n_processes=args.number_of_processes,
            n_instances=total_instances,
        )

        for scarcity in scarcities:
            print(
                f"Generating {args.n_instances} instance(s) "
                f"for scarcity {scarcity:.1f}"
            )

            for _ in range(args.n_instances):
                generator_params = {
                    key: value for key, value in params.items() if key != "timeout"
                }

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
                    generation_metadata={
                        "dataset_id": dataset_id,
                        "dataset_scarcity": scarcity,
                    },
                )

                data_dir = f"data/{instance_id}"
                os.makedirs(data_dir, exist_ok=True)

                processes_file = f"{data_dir}/processes.pkl"
                save_pickle(processes_file, (processes, global_resource_ids))
                db.update_instance_processes_file(instance_id, processes_file)

                db.add_instance_to_dataset(instance_id, dataset_id)

                for model_name in models:
                    db.create_pending_solution(
                        instance_id=instance_id,
                        solver=model_name,
                        scarcity=scarcity,
                        objective=args.objective,
                    )

                print(
                    f"Created instance {instance_id} "
                    f"for scarcity {scarcity:.1f}"
                )

        print(f"Created dataset {dataset_id}")
        print(f"Total instances: {total_instances}")
        print(f"Instances per scarcity: {args.n_instances}")


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


def add_generator_arguments(parser: argparse.ArgumentParser) -> None:
    for key, default in DEFAULT_PARAMS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}",
            type=ARG_TYPES.get(key, type(default)),
            default=default,
        )


def add_xml_files_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--xml-files",
        nargs="+",
        default=None,
        dest="xml_files",
        metavar="FILE[,FILE...]",
        help=(
            "RA-PST XML file(s) per phase. Each argument corresponds to one "
            "phase, cycled if fewer than --max-phases are given. Use a "
            "comma-separated list to define a per-phase pool, e.g. "
            '--xml-files phase0.xml "phase1a.xml,phase1b.xml"'
        ),
    )


def add_objective_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--objective",
        default="flow-time",
        choices=["makespan", "flow-time"],
        help=(
            "Objective used for newly created pending rows, or as fallback "
            "for legacy solution rows where objective is NULL."
        ),
    )


def add_models_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=None,
        metavar="MODEL",
        help=f"Subset of models. Default: all. Choices: {MODELS}",
    )


def add_scarcities_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scarcities",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
        help="Subset of scarcity levels. Default: all configured scarcities.",
    )


def add_force_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Resolve existing solution rows even if they are already solved.",
    )


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

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    gen = sub.add_parser(
        "generate",
        help="Generate a single instance",
    )
    add_generator_arguments(gen)
    add_xml_files_argument(gen)
    add_objective_argument(gen)
    add_models_argument(gen)
    add_scarcities_argument(gen)
    gen.add_argument(
        "--create-pending",
        action="store_true",
        help=(
            "Also create pending solution rows for the selected models and "
            "scarcities. By default, generate only creates the instance."
        ),
    )
    gen.set_defaults(func=cmd_generate)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    run = sub.add_parser(
        "run",
        help="Solve unsolved database entries for one instance",
    )
    run.add_argument(
        "instance_id",
        type=int,
        help="Instance ID in the database",
    )
    add_objective_argument(run)
    add_force_argument(run)
    run.set_defaults(func=cmd_run)

    # ------------------------------------------------------------------
    # run-all
    # ------------------------------------------------------------------

    run_all = sub.add_parser(
        "run-all",
        help="Solve all unsolved database entries",
    )
    add_objective_argument(run_all)
    add_force_argument(run_all)
    run_all.set_defaults(func=cmd_run_all)

    # ------------------------------------------------------------------
    # run-dataset
    # ------------------------------------------------------------------

    run_dataset = sub.add_parser(
        "run-dataset",
        aliases=["run-all-in-dataset"],
        help="Solve all unsolved database entries in a dataset",
    )
    run_dataset.add_argument(
        "dataset_id",
        type=int,
        help="Dataset ID in the database",
    )
    add_objective_argument(run_dataset)
    add_force_argument(run_dataset)
    run_dataset.set_defaults(func=cmd_run_dataset)

    # ------------------------------------------------------------------
    # create-dataset
    # ------------------------------------------------------------------

    create_dataset = sub.add_parser(
        "create-dataset",
        help=(
            "Create a dataset. --n-instances is interpreted as the number of "
            "instances to generate per scarcity."
        ),
    )
    create_dataset.add_argument("name", type=str, help="Dataset name")
    create_dataset.add_argument(
        "--description",
        type=str,
        default="",
        help="Dataset description",
    )
    add_generator_arguments(create_dataset)
    create_dataset.add_argument(
        "--n-instances",
        dest="n_instances",
        type=int,
        default=10,
        help="Number of instances to generate per scarcity",
    )
    add_scarcities_argument(create_dataset)
    add_models_argument(create_dataset)
    add_objective_argument(create_dataset)
    add_xml_files_argument(create_dataset)
    create_dataset.set_defaults(func=cmd_create_dataset)

    # ------------------------------------------------------------------
    # table
    # ------------------------------------------------------------------

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
    add_models_argument(table)
    add_scarcities_argument(table)
    table.set_defaults(func=cmd_table)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
