from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from gurobipy import GRB


class Database:
    def __init__(self, filename: str):
        self.filename = filename
        self.conn = sqlite3.connect(filename)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        self._create_tables()
        self._migrate()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS instances (
                id                    INTEGER PRIMARY KEY,
                number_of_processes   INTEGER,
                arrival_rate          REAL,
                batch_size            REAL,
                max_phases            INTEGER,
                min_base_duration     REAL,
                max_base_duration     REAL,
                min_resource_ratio    REAL,
                resource_ratio_center REAL,
                resource_ratio_spread REAL,
                timeout               INTEGER,
                n_resources           INTEGER,
                processes_file        TEXT,
                seed                  INTEGER,
                xml_files             TEXT,
                global_resource_ids   TEXT,
                generator_version     TEXT,
                generation_metadata   TEXT
            )
            """
        )

        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                id            INTEGER PRIMARY KEY,
                instance_id   INTEGER NOT NULL,
                scarcity      REAL NOT NULL,
                instance_file TEXT,
                T             INTEGER,
                n_jobs        INTEGER,
                n_resources   INTEGER,
                capacities    TEXT,
                descriptors   TEXT,
                UNIQUE (instance_id, scarcity),
                FOREIGN KEY (instance_id) REFERENCES instances (id)
            )
            """
        )

        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS solution (
                id             INTEGER PRIMARY KEY,
                instance_id    INTEGER,
                solver         TEXT,
                sol_file       TEXT,
                instance_file  TEXT,
                scarcity       REAL,
                divisor        INTEGER,
                solved         BOOLEAN,
                status         TEXT,
                objective      TEXT,
                objective_val  REAL,
                scenario_id    INTEGER,
                finished       BOOLEAN,
                feasible       BOOLEAN,
                optimal        BOOLEAN,
                best_bound     REAL,
                mip_gap        REAL,
                runtime        REAL,
                FOREIGN KEY (scenario_id) REFERENCES scenarios (id)
            )
            """
        )

        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id            INTEGER PRIMARY KEY,
                name          TEXT,
                description   TEXT,
                n_processes   INTEGER,
                n_instances   INTEGER
            )
            """
        )

        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS instance_dataset (
                id            INTEGER PRIMARY KEY,
                instance_id   INTEGER,
                dataset_id    INTEGER,
                FOREIGN KEY (instance_id) REFERENCES instances (id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                UNIQUE (instance_id, dataset_id)
            )
            """
        )

        self.conn.commit()

    def _table_columns(self, table: str) -> set[str]:
        self.cur.execute(f"PRAGMA table_info({table})")
        return {row["name"] for row in self.cur.fetchall()}

    def _add_column_if_missing(self, table: str, column: str, definition: str) -> None:
        if column not in self._table_columns(table):
            self.cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _migrate(self) -> None:
        for col, definition in {
            "seed": "INTEGER",
            "xml_files": "TEXT",
            "global_resource_ids": "TEXT",
            "generator_version": "TEXT",
            "generation_metadata": "TEXT",
        }.items():
            self._add_column_if_missing("instances", col, definition)

        for col, definition in {
            "scenario_id": "INTEGER",
            "finished": "BOOLEAN",
            "feasible": "BOOLEAN",
            "optimal": "BOOLEAN",
            "best_bound": "REAL",
            "mip_gap": "REAL",
            "runtime": "REAL",
        }.items():
            self._add_column_if_missing("solution", col, definition)

        for col, definition in {
            "n_jobs": "INTEGER",
            "n_resources": "INTEGER",
        }.items():
            self._add_column_if_missing("scenarios", col, definition)

        self.conn.commit()

    # ------------------------------------------------------------------
    # Instances
    # ------------------------------------------------------------------

    def add_instance(
        self,
        number_of_processes: int,
        arrival_rate: float,
        batch_size: float,
        max_phases: int,
        min_base_duration: float,
        max_base_duration: float,
        min_resource_ratio: float,
        resource_ratio_center: float,
        resource_ratio_spread: float,
        timeout: int,
        n_resources: int = 0,
        processes_file: str = "",
        seed: Optional[int] = None,
        xml_files: Optional[list[str | list[str]]] = None,
        global_resource_ids: Optional[list[str]] = None,
        generator_version: str = "v2",
        generation_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        self.cur.execute(
            """
            INSERT INTO instances (
                number_of_processes,
                arrival_rate,
                batch_size,
                max_phases,
                min_base_duration,
                max_base_duration,
                min_resource_ratio,
                resource_ratio_center,
                resource_ratio_spread,
                timeout,
                n_resources,
                processes_file,
                seed,
                xml_files,
                global_resource_ids,
                generator_version,
                generation_metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                number_of_processes,
                arrival_rate,
                batch_size,
                max_phases,
                min_base_duration,
                max_base_duration,
                min_resource_ratio,
                resource_ratio_center,
                resource_ratio_spread,
                timeout,
                n_resources,
                processes_file,
                seed,
                json.dumps(xml_files),
                json.dumps(global_resource_ids),
                generator_version,
                json.dumps(generation_metadata or {}),
            ),
        )
        self.conn.commit()
        return int(self.cur.lastrowid)

    def update_instance_processes_file(self, instance_id: int, processes_file: str) -> None:
        self.cur.execute(
            "UPDATE instances SET processes_file = ? WHERE id = ?",
            (processes_file, instance_id),
        )
        self.conn.commit()

    def get_instance(self, instance_id: int) -> sqlite3.Row | None:
        self.cur.execute("SELECT * FROM instances WHERE id = ?", (instance_id,))
        return self.cur.fetchone()

    def get_instances(self) -> list[sqlite3.Row]:
        self.cur.execute("SELECT * FROM instances")
        return self.cur.fetchall()

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------

    def upsert_scenario(
        self,
        instance_id: int,
        scarcity: float,
        instance_file: str,
        T: int,
        n_jobs: int,
        n_resources: int,
        capacities: list[int],
        descriptors: Optional[dict[str, Any]] = None,
    ) -> int:
        self.cur.execute(
            """
            INSERT INTO scenarios (
                instance_id,
                scarcity,
                instance_file,
                T,
                n_jobs,
                n_resources,
                capacities,
                descriptors
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instance_id, scarcity)
            DO UPDATE SET
                instance_file = excluded.instance_file,
                T             = excluded.T,
                n_jobs        = excluded.n_jobs,
                n_resources   = excluded.n_resources,
                capacities    = excluded.capacities,
                descriptors   = excluded.descriptors
            """,
            (
                instance_id,
                scarcity,
                instance_file,
                T,
                n_jobs,
                n_resources,
                json.dumps(capacities),
                json.dumps(descriptors or {}),
            ),
        )
        self.conn.commit()
        row = self.get_scenario(instance_id, scarcity)
        if row is None:
            raise RuntimeError("Failed to create or update scenario")
        return int(row["id"])

    def get_scenario(self, instance_id: int, scarcity: float) -> sqlite3.Row | None:
        self.cur.execute(
            """
            SELECT * FROM scenarios
            WHERE instance_id = ? AND scarcity = ?
            """,
            (instance_id, scarcity),
        )
        return self.cur.fetchone()

    def get_scenario_by_id(self, scenario_id: int) -> sqlite3.Row | None:
        self.cur.execute("SELECT * FROM scenarios WHERE id = ?", (scenario_id,))
        return self.cur.fetchone()

    # ------------------------------------------------------------------
    # Solutions
    # ------------------------------------------------------------------

    def add_solution(
        self,
        instance_id: int,
        solver: str,
        sol_file: str,
        instance_file: str,
        scarcity: float,
        divisor: int,
        solved: bool,
        status: str,
        objective: str,
        objective_val: float | None,
        scenario_id: int | None = None,
        finished: bool | None = None,
        feasible: bool | None = None,
        optimal: bool | None = None,
        best_bound: float | None = None,
        mip_gap: float | None = None,
        runtime: float | None = None,
    ) -> int:
        self.cur.execute(
            """
            INSERT INTO solution (
                instance_id,
                solver,
                sol_file,
                instance_file,
                scarcity,
                divisor,
                solved,
                status,
                objective,
                objective_val,
                scenario_id,
                finished,
                feasible,
                optimal,
                best_bound,
                mip_gap,
                runtime
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                instance_id,
                solver,
                sol_file,
                instance_file,
                scarcity,
                divisor,
                solved,
                status,
                objective,
                objective_val,
                scenario_id,
                finished,
                feasible,
                optimal,
                best_bound,
                mip_gap,
                runtime,
            ),
        )
        self.conn.commit()
        return int(self.cur.lastrowid)

    def update_solution(
        self,
        instance_id: int,
        solver: str,
        scarcity: float,
        sol_file: str,
        instance_file: str,
        divisor: int,
        solved: bool,
        status: str,
        objective: str,
        objective_val: float | None,
        scenario_id: int | None = None,
        finished: bool | None = None,
        feasible: bool | None = None,
        optimal: bool | None = None,
        best_bound: float | None = None,
        mip_gap: float | None = None,
        runtime: float | None = None,
    ) -> None:
        self.cur.execute(
            """
            UPDATE solution
            SET
                sol_file      = ?,
                instance_file = ?,
                divisor       = ?,
                solved        = ?,
                status        = ?,
                objective     = ?,
                objective_val = ?,
                scenario_id   = ?,
                finished      = ?,
                feasible      = ?,
                optimal       = ?,
                best_bound    = ?,
                mip_gap       = ?,
                runtime       = ?
            WHERE instance_id = ? AND solver = ? AND scarcity = ?
            """,
            (
                sol_file,
                instance_file,
                divisor,
                solved,
                status,
                objective,
                objective_val,
                scenario_id,
                finished,
                feasible,
                optimal,
                best_bound,
                mip_gap,
                runtime,
                instance_id,
                solver,
                scarcity,
            ),
        )
        self.conn.commit()

    def get_solution(
        self,
        instance_id: int,
        solver: str,
        scarcity: float,
    ) -> sqlite3.Row | None:
        self.cur.execute(
            """
            SELECT * FROM solution
            WHERE instance_id = ? AND solver = ? AND scarcity = ?
            """,
            (instance_id, solver, scarcity),
        )
        return self.cur.fetchone()

    def get_solutions(self, instance_id: int) -> list[sqlite3.Row]:
        self.cur.execute(
            "SELECT * FROM solution WHERE instance_id = ?",
            (instance_id,),
        )
        return self.cur.fetchall()

    def record_solution(
        self,
        *,
        instance_id: int,
        scenario_id: int | None,
        solver: str,
        sol_file: str,
        instance_file: str,
        scarcity: float,
        divisor: int,
        objective: str,
        status: int,
        finished: bool,
        feasible: bool,
        optimal: bool,
        objective_val: float | None,
        best_bound: float | None,
        mip_gap: float | None,
        runtime: float | None,
    ) -> int:
        row = self.get_solution(instance_id, solver, scarcity)
        kwargs = dict(
            instance_id=instance_id,
            solver=solver,
            sol_file=sol_file,
            instance_file=instance_file,
            scarcity=scarcity,
            divisor=divisor,
            solved=feasible,
            status=str(status),
            objective=objective,
            objective_val=objective_val,
            scenario_id=scenario_id,
            finished=finished,
            feasible=feasible,
            optimal=optimal,
            best_bound=best_bound,
            mip_gap=mip_gap,
            runtime=runtime,
        )
        if row is None:
            return self.add_solution(**kwargs)
        self.update_solution(**kwargs)
        return int(row["id"])

    def create_pending_solution(
        self,
        instance_id: int,
        solver: str,
        scarcity: float,
        objective: str,
    ) -> int:
        row = self.get_solution(instance_id, solver, scarcity)
        if row is not None:
            return int(row["id"])

        self.cur.execute(
            """
            INSERT INTO solution (
                instance_id,
                solver,
                scarcity,
                objective,
                solved
            )
            VALUES (?, ?, ?, ?, 0)
            """,
            (instance_id, solver, scarcity, objective),
        )
        self.conn.commit()
        return int(self.cur.lastrowid)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        description: str,
        n_processes: int,
        n_instances: int,
    ) -> int:
        self.cur.execute(
            """
            INSERT INTO datasets (
                name,
                description,
                n_processes,
                n_instances
            )
            VALUES (?, ?, ?, ?)
            """,
            (name, description, n_processes, n_instances),
        )
        self.conn.commit()
        return int(self.cur.lastrowid)

    def get_datasets(self) -> list[sqlite3.Row]:
        self.cur.execute("SELECT * FROM datasets")
        return self.cur.fetchall()

    def get_dataset(self, dataset_id: int) -> list[sqlite3.Row]:
        self.cur.execute(
            """
            SELECT instances.*
            FROM instances
            INNER JOIN instance_dataset
                ON instances.id = instance_dataset.instance_id
            WHERE instance_dataset.dataset_id = ?
            """,
            (dataset_id,),
        )
        return self.cur.fetchall()

    def add_instance_to_dataset(self, instance_id: int, dataset_id: int) -> None:
        self.cur.execute(
            """
            SELECT * FROM instance_dataset
            WHERE instance_id = ? AND dataset_id = ?
            """,
            (instance_id, dataset_id),
        )
        if self.cur.fetchone() is not None:
            raise ValueError(
                f"Instance {instance_id} is already contained in dataset {dataset_id}"
            )

        self.cur.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        dataset = self.cur.fetchone()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        self.cur.execute("SELECT * FROM instances WHERE id = ?", (instance_id,))
        instance = self.cur.fetchone()
        if instance is None:
            raise ValueError(f"Instance {instance_id} not found")

        if instance["number_of_processes"] != dataset["n_processes"]:
            raise ValueError(
                f"Instance {instance_id} has {instance['number_of_processes']} "
                f"processes, dataset requires {dataset['n_processes']}"
            )

        self.cur.execute(
            """
            INSERT INTO instance_dataset (instance_id, dataset_id)
            VALUES (?, ?)
            """,
            (instance_id, dataset_id),
        )
        self.conn.commit()

    def get_all_instance_to_dataset(self) -> list[sqlite3.Row]:
        self.cur.execute("SELECT * FROM instance_dataset")
        return self.cur.fetchall()

    def get_unsolved_solutions_for_dataset(self, dataset_id: int) -> list[sqlite3.Row]:
        self.cur.execute(
            """
            SELECT solution.*
            FROM solution
            INNER JOIN instance_dataset
                ON solution.instance_id = instance_dataset.instance_id
            WHERE instance_dataset.dataset_id = ? AND NOT solution.solved
            """,
            (dataset_id,),
        )
        return self.cur.fetchall()

    # ------------------------------------------------------------------
    # LaTeX tables
    # ------------------------------------------------------------------

    def _latex_table_header(
        self,
        caption: str,
        label: str,
        descriptor: str,
        scarcities: list[float],
    ) -> str:
        columns = "|".join("c" for _ in scarcities)
        scarcity_row = " & ".join(f"{s:.1f}" for s in scarcities)
        caption_text = f"{caption} ({descriptor})" if caption else descriptor
        return (
            "\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption_text}. "
            "An entry with - indicates an infeasible instance.}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{|l|{columns}|}}\n"
            "\\hline\n"
            f"\\(RS\\) & {scarcity_row} \\\\ \\hline\n"
        )

    def _solution_info(self, sol_file: str | None) -> dict[str, Any] | None:
        if not sol_file:
            return None
        try:
            with open(sol_file, "r") as f:
                data = json.load(f)
            return data.get("SolutionInfo")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def make_latex_tables(
        self,
        instance_id: int,
        models: list[str],
        scarcities: list[float],
    ) -> tuple[str, str]:
        instance = self.get_instance(instance_id)
        if instance is None:
            raise ValueError(f"Instance {instance_id} not found")

        n_proc = instance["number_of_processes"]
        footer = "\\end{tabular}\n\\end{table}\n"

        objective_table = self._latex_table_header(
            "Makespan",
            "tab:makespan",
            f"{n_proc} processes",
            scarcities,
        )
        runtime_table = self._latex_table_header(
            "Runtime",
            "tab:runtime",
            f"{n_proc} processes",
            scarcities,
        )

        for model in models:
            obj_row = model
            rt_row = model

            for scarcity in scarcities:
                sol = self.get_solution(instance_id, model, scarcity)
                info = self._solution_info(sol["sol_file"] if sol else None)

                if info is None:
                    obj_row += " & ?"
                    rt_row += " & ?"
                    continue

                status = info["Status"]
                sol_count = info["SolCount"]

                if status == GRB.INFEASIBLE or (
                    sol_count == 0 and status == GRB.TIME_LIMIT
                ):
                    obj_row += " & -"
                    rt_row += " & -"
                else:
                    obj = int(info["ObjVal"])
                    obj_cell = (
                        f"\\textbf{{{obj}}}" if status == GRB.OPTIMAL else str(obj)
                    )
                    obj_row += f" & {obj_cell}"
                    rt_row += f" & {info['Runtime']:.1f}s"

            objective_table += obj_row + " \\\\ \\hline\n"
            runtime_table += rt_row + " \\\\ \\hline\n"

        return objective_table + footer, runtime_table + footer

    def make_dataset_latex_tables(
        self,
        dataset_id: int,
        models: list[str],
        scarcities: list[float],
    ) -> tuple[str, str]:
        self.cur.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        dataset = self.cur.fetchone()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        descriptor = (
            f"{dataset['n_processes']} processes, {dataset['n_instances']} instances"
        )
        instance_ids = [row["id"] for row in self.get_dataset(dataset_id)]

        footer = "\\end{tabular}\n\\end{table}\n"
        objective_table = self._latex_table_header(
            "",
            "tab:makespan",
            descriptor,
            scarcities,
        )
        runtime_table = self._latex_table_header(
            "Mean runtime (s)",
            "tab:runtime",
            descriptor,
            scarcities,
        )

        for model in models:
            obj_row = model
            rt_row = model

            for scarcity in scarcities:
                obj_vals: list[float] = []
                runtimes: list[float] = []
                n_missing = 0

                for iid in instance_ids:
                    sol = self.get_solution(iid, model, scarcity)
                    info = self._solution_info(sol["sol_file"] if sol else None)

                    if info is None:
                        n_missing += 1
                        continue

                    status = info["Status"]
                    sol_count = info["SolCount"]
                    if status == GRB.INFEASIBLE or (
                        sol_count == 0 and status == GRB.TIME_LIMIT
                    ):
                        continue

                    obj_vals.append(float(info["ObjVal"]))
                    runtimes.append(float(info["Runtime"]))

                if n_missing == len(instance_ids):
                    obj_row += " & ?"
                    rt_row += " & ?"
                elif not obj_vals:
                    obj_row += " & -"
                    rt_row += " & -"
                else:
                    obj_row += f" & {sum(obj_vals) / len(obj_vals):.0f}"
                    rt_row += f" & {sum(runtimes) / len(runtimes):.1f}"

            objective_table += obj_row + " \\\\ \\hline\n"
            runtime_table += rt_row + " \\\\ \\hline\n"

        return objective_table + footer, runtime_table + footer
