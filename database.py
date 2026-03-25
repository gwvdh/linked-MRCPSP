import sqlite3
import json
import os


class Database:
    def __init__(self, filename: str):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS instances 
                (
                id INTEGER PRIMARY KEY, 
                number_of_processes INTEGER,
                arrival_rate REAL,
                batch_size REAL,
                max_phases INTEGER, 
                min_base_duration REAL, 
                max_base_duration REAL, 
                min_resource_1_ratio REAL, 
                resource_1_ratio_center REAL, 
                resource_1_ratio_spread REAL, 
                min_resource_2_ratio REAL, 
                resource_2_ratio_center REAL, 
                resource_2_ratio_spread REAL, 
                res_1_2_multiplier REAL, 
                res_1_3_multiplier REAL, 
                job_3_multiplier REAL
                )
            """)
        self.cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS solution
                (
                id INTEGER PRIMARY KEY, 
                instance_id INTEGER, 
                solver TEXT,
                sol_file TEXT, 
                instance_file TEXT,
                scarcity REAL, 
                divisor INTEGER,
                solved BOOLEAN,
                status TEXT,
                objective TEXT,
                objective_val REAL
                )
            """)
        self.conn.commit()

    def __exit__(self, *args): 
        self.conn.close()

    def add_instance(self, 
                     number_of_processes: int, 
                     arrival_rate: float, 
                     batch_size: float,
                     max_phases: int, 
                     min_base_duration: float, 
                     max_base_duration: float, 
                     min_resource_1_ratio: float, 
                     resource_1_ratio_center: float, 
                     resource_1_ratio_spread: float, 
                     min_resource_2_ratio: float, 
                     resource_2_ratio_center: float, 
                     resource_2_ratio_spread: float, 
                     res_1_2_multiplier: float, 
                     res_1_3_multiplier: float, 
                     job_3_multiplier: float
                     ):
        sql = '''
            INSERT INTO instances (number_of_processes, arrival_rate, batch_size, max_phases, min_base_duration, max_base_duration, min_resource_1_ratio, resource_1_ratio_center, resource_1_ratio_spread, min_resource_2_ratio, resource_2_ratio_center, resource_2_ratio_spread, res_1_2_multiplier, res_1_3_multiplier, job_3_multiplier) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(sql, (
            number_of_processes, 
            arrival_rate, 
            batch_size,
            max_phases, 
            min_base_duration, 
            max_base_duration, 
            min_resource_1_ratio, 
            resource_1_ratio_center, 
            resource_1_ratio_spread, 
            min_resource_2_ratio, 
            resource_2_ratio_center, 
            resource_2_ratio_spread, 
            res_1_2_multiplier, 
            res_1_3_multiplier, 
            job_3_multiplier
        ))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_instances(self):
        self.cursor.execute("SELECT * FROM instances")
        return self.cursor.fetchall()

    def get_instance(self, id: int):
        self.cursor.execute("SELECT * FROM instances WHERE id = ?", (id,))
        return self.cursor.fetchone()

    def add_solution(self, 
                     instance_id: int, 
                     solver: str, 
                     sol_file: str, 
                     instance_file: str, 
                     scarcity: float, 
                     divisor: int, 
                     solved: bool, 
                     status: str, 
                     objective: str, 
                     objective_val: float
                     ):
        sql = '''
            INSERT INTO solution (instance_id, solver, sol_file, instance_file, scarcity, divisor, solved, status, objective, objective_val) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(sql, (
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
        ))
        self.conn.commit()
        return self.cursor.lastrowid

    def update_solution(self, 
                        instance_id: int, 
                        solver: str, 
                        sol_file: str, 
                        instance_file: str, 
                        scarcity: float, 
                        divisor: int, 
                        solved: bool, 
                        status: str, 
                        objective: str, 
                        objective_val: float
                        ):
        sql = '''
            UPDATE solution SET 
            solver = ?, 
            sol_file = ?, 
            instance_file = ?, 
            scarcity = ?, 
            divisor = ?, 
            solved = ?, 
            status = ?, 
            objective = ?, 
            objective_val = ?
            WHERE instance_id = ? AND solver = ? AND scarcity = ?
        '''
        self.cursor.execute(sql, (
            solver, 
            sol_file, 
            instance_file, 
            scarcity, 
            divisor, 
            solved, 
            status, 
            objective, 
            objective_val,
            instance_id,
            solver,
            scarcity
        ))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_solutions(self, instance_id):
        self.cursor.execute("SELECT * FROM solution WHERE instance_id = ?", (instance_id,))
        return self.cursor.fetchall()

    def get_solution(self, instance_id, model_name, scarcity):
        self.cursor.execute("SELECT * FROM solution WHERE instance_id = ? AND solver = ? AND scarcity = ?", (instance_id, model_name, scarcity))
        return self.cursor.fetchone()

    def close(self):
        self.conn.close()



