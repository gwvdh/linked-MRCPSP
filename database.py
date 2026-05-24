import sqlite3


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
