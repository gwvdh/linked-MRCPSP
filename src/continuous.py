import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from .utils import normalize
from .model import GurobiModel
from .vis_schedule import visualize_continuous_model


class ContinuousModel(GurobiModel):
    def initialize_model(self):
        # Initialize model
        self.model = gp.Model("continuous")
        if self.silent:
            self.model.setParam('OutputFlag', False)
        self.model.setParam('TimeLimit', self.timeout)

        # Activity variables
        S = self.model.addVars(self.n, vtype=GRB.INTEGER, name="activity")
        self.model.addConstrs((S[i] >= self.ES[i] for i in range(self.n)), name="earliest_starting_times")
        self.model.addConstrs((S[i] <= self.LS[i] for i in range(self.n)), name="latest_starting_times")
        activity_mode_sets = [(i, m) for i in range(self.n) for m in range(self.M[i])]
        self.x = self.model.addVars(activity_mode_sets, vtype=GRB.BINARY, name="mode")
        pair_sets = [(i, j) for i, j in self.VP]
        self.y = self.model.addVars(pair_sets, vtype=GRB.BINARY, name="completion_start_sequence")
        pair_sets_2 = [(i, j) for i, j in self.VP]
        self.z = self.model.addVars(pair_sets_2, vtype=GRB.BINARY, name="start_start_sequence")
        #resource_sets = [(i, j, k) for i in range(self.n) for j in range(self.n) for k in range(len(self.R))]
        resource_sets = [(j, i, k) for j, i in self.VP for k in range(len(self.R))]
        self.u = self.model.addVars(resource_sets, vtype=GRB.INTEGER, lb=0, ub={(j, i, k): max(self.r[j][m][k] for m in range(self.M[j])) for j, i, k in resource_sets}, name="resource")

        # Objective
        if self.obj == "makespan":
            self.model.setObjective(S[self.n-1], GRB.MINIMIZE)
        elif self.obj == "flow-time":
            self.model.setObjective(gp.quicksum(S[i] - self.ES[i] + gp.quicksum(self.p[i][m] * self.x[i, m] for m in range(self.M[i])) for i in range(self.n)), GRB.MINIMIZE)
        elif self.obj == "process-flow-time":
            self.model.setObjective(gp.quicksum(S[i] - self.ES[i] + gp.quicksum(self.p[i][m] * self.x[i, m] for m in range(self.M[i])) for i in self.O), GRB.MINIMIZE)

        # Constraints
        # Connect the start-time variables with the completion-start sequencing variables
        self.model.addConstrs((S[i] + gp.quicksum(self.p[i][m] * self.x[i, m] for m in range(self.M[i])) <= S[j] + self.T*(1-self.y[i, j]) for i,j in self.VP), name="connect_start_completion")

        # Precedence relations between jobs (i,j)
        self.model.addConstrs((S[i] + gp.quicksum(self.p[i][m] * self.x[i, m] for m in range(self.M[i])) <= S[j] for i,j in self.E), name="precedence")

        # Connect the start-start sequencing variables and the timing variables
        self.model.addConstrs((self.T * self.z[i, j] >= S[j] - S[i] + 1 for i,j in self.VP), name="connect_start_start_timing")

        # Execute each activity in exactly one mode
        self.model.addConstrs((gp.quicksum(self.x[i, m] for m in range(self.M[i])) == 1 for i in range(self.n)), name="execute_activity")

        # Resource availability
        BIG_M = self.T 
        self.model.addConstrs((gp.quicksum(self.r[j][m][k] * self.x[j, m] for m in range(self.M[j])) - BIG_M*(1 - self.z[j, i] + self.y[j, i]) - BIG_M*gp.quicksum(self.x[i, m] for m in range(self.M[i]) if self.p[i][m] == 0) <= self.u[j,i,k] for j, i in self.VP for k in range(len(self.R))), name="resource")
        self.model.addConstrs((gp.quicksum(self.r[i][m][k] * self.x[i, m] for m in range(self.M[i])) + gp.quicksum(self.u[j, i, k] for j, i_prime in self.VP if i_prime == i) <= self.R[k] for i in range(self.n) for k in range(len(self.R))), name="resource_2")

        # Linked modes of jobs (i,j)
        self.model.addConstrs((self.x[i, m] == self.x[j, m] for i,j in self.L for m in range(self.M[i])), name="linked")
        
        return self.model

    def visualize(self, filename: str) -> None:
        visualize_continuous_model(self.model, self.n, self.T, self.M, self.R, self.p, self.r, self.processes, self.divisor, activity_names=None, filename=filename)


