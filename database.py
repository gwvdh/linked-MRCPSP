import sqlite3
import json


class Database:
    def __init__(self, filename: str):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS instances
            (
                id                   INTEGER PRIMARY KEY,
                number_of_processes  INTEGER,
                arrival_rate         REAL,
                batch_size           REAL,
                max_phases           INTEGER,
                min_base_duration    REAL,
                max_base_duration    REAL,
                min_resource_ratio   REAL,
                resource_ratio_center REAL,
                resource_ratio_spread REAL,
                timeout              INTEGER
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS solution
            (
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
    ) -> int:
        sql = """
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
                timeout
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(
            sql,
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
            ),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_instance(self, id: int):
        self.cursor.execute("SELECT * FROM instances WHERE id = ?", (id,))
        return self.cursor.fetchone()

    def get_instances(self):
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
        sql = """
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
        """
        self.cursor.execute(
            sql,
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
        sql = """
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
        """
        self.cursor.execute(
            sql,
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

    def get_solution(self, instance_id: int, model_name: str, scarcity: float):
        self.cursor.execute(
            """
            SELECT * FROM solution
            WHERE instance_id = ? AND solver = ? AND scarcity = ?
            """,
            (instance_id, model_name, scarcity),
        )
        return self.cursor.fetchone()

    def get_solutions(self, instance_id: int):
        self.cursor.execute(
            "SELECT * FROM solution WHERE instance_id = ?", (instance_id,)
        )
        return self.cursor.fetchall()
