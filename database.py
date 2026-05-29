import sqlite3
import json
from gurobipy import GRB


class Database:
    def __init__(self, filename: str):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self.conn.row_factory = sqlite3.Row  # column access by name
        self.cursor = self.conn.cursor()
        self.cursor.execute(
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
                processes_file        TEXT
            )
            """
        )
        self.cursor.execute(
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
                objective_val  REAL
            )
            """
        )
        self.cursor.execute(
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
        self.cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS instance_dataset (
                    id            INTEGER PRIMARY KEY,
                    instance_id   INTEGER,
                    dataset_id    INTEGER,
                    FOREIGN KEY (instance_id) REFERENCES instances (id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            """
        )
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.conn.close()

    def close(self):
        self.conn.close()

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
    ) -> int:
        self.cursor.execute(
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
                processes_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def update_instance_processes_file(
        self, id: int, processes_file: str
    ) -> None:
        self.cursor.execute(
            "UPDATE instances SET processes_file = ? WHERE id = ?",
            (processes_file, id),
        )
        self.conn.commit()

    def get_instance(self, id: int) -> sqlite3.Row | None:
        self.cursor.execute("SELECT * FROM instances WHERE id = ?", (id,))
        return self.cursor.fetchone()

    def get_instances(self) -> list[sqlite3.Row]:
        self.cursor.execute("SELECT * FROM instances")
        return self.cursor.fetchall()

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
    ) -> int:
        self.cursor.execute(
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
                objective_val
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def update_solution(
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
    ) -> int:
        self.cursor.execute(
            """
            UPDATE solution
            SET
                sol_file      = ?,
                instance_file = ?,
                divisor       = ?,
                solved        = ?,
                status        = ?,
                objective     = ?,
                objective_val = ?
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
                instance_id,
                solver,
                scarcity,
            ),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_solution(
        self, instance_id: int, model_name: str, scarcity: float
    ) -> sqlite3.Row | None:
        self.cursor.execute(
            """
            SELECT * FROM solution
            WHERE instance_id = ? AND solver = ? AND scarcity = ?
            """,
            (instance_id, model_name, scarcity),
        )
        return self.cursor.fetchone()

    def get_solutions(self, instance_id: int) -> list[sqlite3.Row]:
        self.cursor.execute(
            "SELECT * FROM solution WHERE instance_id = ?", (instance_id,)
        )
        return self.cursor.fetchall()

    def create_pending_solution(
        self, instance_id: int, solver: str, scarcity: float, objective:str
    ) -> int:
        self.cursor.execute(
            """
            INSERT INTO solution (instance_id, solver, scarcity, objective, solved)
            VALUES (?, ?, ?, ?, 0)
            """,
            (instance_id, solver, scarcity, objective),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_unsolved_solutions_for_dataset(
        self, dataset_id: int
    ) -> list[sqlite3.Row]:
        self.cursor.execute(
            """
            SELECT solution.* FROM solution
            INNER JOIN instance_dataset ON solution.instance_id = instance_dataset.instance_id
            WHERE instance_dataset.dataset_id = ? AND NOT solution.solved
            """,
            (dataset_id,),
        )
        return self.cursor.fetchall()

    def get_datasets(self) -> list[sqlite3.Row]:
        self.cursor.execute("SELECT * FROM datasets")
        return self.cursor.fetchall()

    def get_dataset(self, id: int) -> list[sqlite3.Row]:
        self.cursor.execute("SELECT * FROM instances INNER JOIN instance_dataset ON instances.id = instance_dataset.instance_id WHERE instance_dataset.dataset_id = ?", (id,))
        return self.cursor.fetchall()

    def add_dataset(self, name: str, description: str, n_processes: int, n_instances: int) -> int:
        self.cursor.execute(
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
        return self.cursor.lastrowid

    def add_instance_to_dataset(self, instance_id: int, dataset_id: int) -> None:
        # Check if the connection exists
        self.cursor.execute("SELECT * FROM instance_dataset WHERE instance_id = ? AND dataset_id = ?", (instance_id, dataset_id))
        if self.cursor.fetchone() is not None:
            raise ValueError(f"Instance {instance_id} already in dataset {dataset_id}")
        # Check if the instance exists
        self.cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        dataset = self.cursor.fetchone()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        number_of_required_processes = dataset["n_processes"]

        self.cursor.execute("SELECT * FROM instances WHERE id = ?", (instance_id,))
        instance = self.cursor.fetchone()
        if instance is None:
            raise ValueError(f"Instance {instance_id} not found")
        if instance["number_of_processes"] != number_of_required_processes:
            raise ValueError(f"Instance {instance_id} has {instance['number_of_processes']} processes, but dataset {dataset_id} requires {number_of_required_processes} processes")
        self.cursor.execute(
            "INSERT INTO instance_dataset (instance_id, dataset_id) VALUES (?, ?)",
            (instance_id, dataset_id),
        )
        self.conn.commit()

    def get_all_instance_to_dataset(self) -> list[sqlite3.Row]:
        self.cursor.execute(
            "SELECT * FROM instance_dataset"
        )
        return self.cursor.fetchall()

    def _latex_table_header(
        self, caption: str, label: str, descriptor: str, scarcities: list[float]
    ) -> str:
        columns = "c|" * len(scarcities)
        scarcity_row = " & ".join(f"{s:.1f}" for s in scarcities)
        return (
            "\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption} ({descriptor}). "
            "An entry with - indicates an infeasible instance.}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{|l|{columns}}}\n"
            f"\\hline\n\\(RS\\) & {scarcity_row} \\\\ \\hline\n"
        )

    def _solution_info(self, sol_file: str | None) -> dict | None:
        if not sol_file:
            return None
        with open(sol_file, "r") as f:
            return json.load(f)["SolutionInfo"]

    def make_latex_tables(
        self, instance_id: int, models: list[str], scarcities: list[float]
    ) -> tuple[str, str]:
        """Make a table of objective and runtime for a given instance and models."""
        n_proc = self.get_instance(instance_id)["number_of_processes"]
        footer = "\\end{tabular}\n\\end{table}\n"
        objective = self._latex_table_header(
                "", "tab:makespan", f"{n_proc} processes", scarcities
            )
        runtime = self._latex_table_header(
                "Runtime", "tab:runtime", f"{n_proc} processes", scarcities
            )
        for model_name in models:
            objective_row = runtime_row = model_name
            for scarcity in scarcities:
                sol = self.get_solution(instance_id, model_name, scarcity)
                info = self._solution_info(sol["sol_file"])
                if info is None:
                    objective_row += " & ?"; runtime_row += " & ?"
                    continue
                status, sol_count = info["Status"], info["SolCount"]
                if status == GRB.INFEASIBLE or (sol_count == 0 and status == GRB.TIME_LIMIT):
                    objective_row += " & -"; runtime_row += " & -"
                else:
                    obj = int(info["ObjVal"])
                    cell = f"\\textbf{{{obj}}}" if status == GRB.OPTIMAL else str(obj)
                    objective_row += f" & {cell}"; runtime_row += f" & {info['Runtime']:.1f}s"
            objective += objective_row + " \\\\ \\hline\n"
            runtime += runtime_row + " \\\\ \\hline\n"
        return objective + footer, runtime + footer

    def make_dataset_latex_tables(
        self, dataset_id: int, models: list[str], scarcities: list[float]
    ) -> tuple[str, str]:
        """Make a table of objective and runtime for all instances in a dataset."""
        self.cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        dataset = self.cursor.fetchone()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        descriptor = f"{dataset['n_processes']} processes, {dataset['n_instances']} instances"
        instance_ids = [row["id"] for row in self.get_dataset(dataset_id)]
        footer = "\\end{tabular}\n\\end{table}\n"
        objective = self._latex_table_header(
            "", "tab:makespan", descriptor, scarcities
        )
        runtime = self._latex_table_header(
            "Mean runtime (s)", "tab:runtime", descriptor, scarcities
        )
        for model_name in models:
            objective_row = runtime_row = model_name
            for scarcity in scarcities:
                obj_vals: list[float] = []
                runtimes: list[float] = []
                n_missing = 0
                for iid in instance_ids:
                    sol = self.get_solution(iid, model_name, scarcity)
                    info = self._solution_info(sol["sol_file"] if sol else None)
                    if info is None:
                        n_missing += 1
                        continue
                    status, sol_count = info["Status"], info["SolCount"]
                    if not (status == GRB.INFEASIBLE or (sol_count == 0 and status == GRB.TIME_LIMIT)):
                        obj_vals.append(int(info["ObjVal"]))
                        runtimes.append(info["Runtime"])
                if n_missing == len(instance_ids):
                    objective_row += " & ?"; runtime_row += " & ?"
                elif not obj_vals:
                    objective_row += " & -"; runtime_row += " & -"
                else:
                    objective_row += f" & {sum(obj_vals) / len(obj_vals):.0f}"
                    runtime_row += f" & {sum(runtimes) / len(runtimes):.1f}"
            objective += objective_row + " \\\\ \\hline\n"
            runtime += runtime_row + " \\\\ \\hline\n"
        return objective + footer, runtime + footer



if __name__ == "__main__":
    _db = Database("database.db")
    print([dict(row) for row in _db.get_instances()])
    print([dict(row) for row in _db.get_datasets()])
    print([dict(row) for row in _db.get_dataset(1)])

    _instances = _db.get_instances()
    _instance_id = 13
    print(f"Instance {_instances[_instance_id]['id']}: {_instances[_instance_id]['number_of_processes']} processes")
    _db.add_instance_to_dataset(_instance_id, 1)
    # after
    print([dict(row) for row in _db.get_datasets()])
    print([dict(row) for row in _db.get_dataset(1)])
    print([dict(row) for row in _db.get_all_instance_to_dataset()])
    _db.close()
