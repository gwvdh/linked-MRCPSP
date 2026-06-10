from ortools.sat.python import cp_model
import gurobipy as gp
from gurobipy import GRB

if __name__ == "__main__":
    from utils import normalize
else:
    from .utils import normalize


class Model:
    def __init__(self, n, T, M, R, E, p, L, r, O, VP, ES=None, LS=None, silent=True, obj="makespan", timeout=600, processes=None):
        """
        n: number of activities
        T: number of time slots 1,...,T
        M: number of modes for each activity
        R: List of resource capacities R[k]
        E: List of pairs of activity indices (i,j) indicating precedence relations
        p: List of processing times for each activity i in each mode m p[i][m]
        L: List of pairs of activity indices (i,j) indicating linked modes
        r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
        O: List of last jobs indices of each process
        VP: List of pairs of activity indices (i,j) indicating the inverse of the precedence relations
        ES: Earliest start time for each activity i
        LS: Latest start time for each activity i
        silent: if True, suppress solver output
        obj: objective function, one of {"makespan", "flow-time"}
        timeout: time limit in seconds
        """
        self.n = n
        self.T = T
        self.M = M
        self.R = R
        self.E = E
        self.p = p
        self.L = L
        self.r = r
        self.O = O
        self.VP = VP
        self.ES = ES
        self.LS = LS
        self.silent = silent
        self.obj = obj
        self.timeout = timeout
        self.processes = processes

        # Normalize processing times
        self.p, self.T, self.divisor = normalize(self.p, self.T)

        # Initialize model
        self.model = self.initialize_model()

    def initialize_model(self):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def is_feasible(self):
        raise NotImplementedError

    def is_optimal(self):
        raise NotImplementedError

    def get_objective(self):
        raise NotImplementedError

    def write(self, filename: str) -> None:
        raise NotImplementedError

    def is_timed_out(self):
        raise NotImplementedError

    def sol_count(self):
        raise NotImplementedError

    def interrupted(self):
        raise NotImplementedError

    def visualize(self, filename: str) -> None:
        raise NotImplementedError

    def solver_time(self):
        raise NotImplementedError

    def lower_bound(self):
        raise NotImplementedError

    def status(self):
        """Gurobi status code. From solver, translate to the Gurobi status codes."""
        raise NotImplementedError

    def cpm_lb(self):
        """
        CPM lower bound derived from ES
        Makespan: ES[n-1]
        Flow-time: sum(min(p[i][m] for m in range(M[i])) for i in range(n))
        """
        if self.obj == "makespan":
            return self.ES[-1]
        elif self.obj == "flow-time":
            return sum(min(self.p[i][m] for m in range(self.M[i])) for i in range(self.n))
        else:
            raise ValueError(f"Unknown objective: {self.obj}")

    def number_of_variables(self):
        raise NotImplementedError

    def number_of_constraints(self):
        raise NotImplementedError

    def number_of_nonzeros(self):
        raise NotImplementedError


class GurobiModel(Model):
    def solve(self):
        return self.model.optimize()

    def update(self):
        return self.model.update()

    def is_feasible(self):
        return self.model.SolCount > 0

    def is_optimal(self):
        return self.model.status == GRB.OPTIMAL

    def get_objective(self):
        if self.model.SolCount == 0:
            return None
        return self.model.objVal * self.divisor

    def write(self, filename: str) -> None:
        self.model.write(filename)

    def is_timed_out(self):
        return self.model.status == GRB.TIME_LIMIT and self.model.SolCount == 0

    def sol_count(self):
        return self.model.SolCount

    def interrupted(self):
        return self.model.status == GRB.INTERRUPTED

    def solver_time(self):
        return self.model.Runtime

    def lower_bound(self):
        return self.model.ObjBound * self.divisor

    def status(self):
        return self.model.status

    def number_of_variables(self):
        return self.model.NumVars

    def number_of_constraints(self):
        return self.model.NumConstrs

    def number_of_nonzeros(self):
        return self.model.NumNZs


class CP_SATModel(Model):
    def solve(self):
        pass

    def update(self):
        pass

    def is_feasible(self):
        pass

    def is_optimal(self):
        pass

    def get_objective(self):
        pass

    def write(self, filename: str) -> None:
        pass

    def is_timed_out(self):
        pass

    def sol_count(self):
        pass

    def interrupted(self):
        pass

    def visualize(self, filename: str) -> None:
        pass

    def solver_time(self):
        pass

    def lower_bound(self):
        pass

    def status(self):
        pass

    def number_of_variables(self):
        pass

    def number_of_constraints(self):
        pass

    def number_of_nonzeros(self):
        pass


