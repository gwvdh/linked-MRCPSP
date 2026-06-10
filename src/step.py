import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from .utils import normalize
from .model import GurobiModel


class StepModel(GurobiModel):
    def initialize_model(self):
        # Initialize model
        self.model = gp.Model("step")
        if self.silent:
            self.model.setParam('OutputFlag', False)
        self.model.setParam('TimeLimit', self.timeout)

        # Step variables
        z_t = {
            i: range(self.ES[i], min(self.LS[i] + 1, self.T))
            for i in range(self.n)
        }
        step_sets = [(i, m, t) for i in range(self.n) for m in range(self.M[i]) for t in range(self.T)]
        self.z = self.model.addVars(step_sets, vtype=GRB.BINARY, name="step")

        # Objective
        if self.obj == "makespan":
            self.model.setObjective(gp.quicksum(t * (self.z[self.n-1, m, t] - self.z[self.n-1, m, t-1]) for t in range(1,self.T) for m in range(self.M[self.n-1])), GRB.MINIMIZE)
        elif self.obj == "flow-time":
            #self.model.setObjective(gp.quicksum((self.z[i, m, t] - self.z[i, m, t-1]) * (t + self.p[i][m] - self.ES[i]) if t > 0 else (self.z[i, m, t]) * (t + self.p[i][m] - self.ES[i]) for i in range(self.n) for m in range(self.M[i]) for t in z_t[i]), GRB.MINIMIZE)
            self.model.setObjective(gp.quicksum(self.p[i][m] + gp.quicksum(1 - (self.z[i, m, t]) for t in z_t[i]) for i in range(self.n) for m in range(self.M[i])), GRB.MINIMIZE)
        elif self.obj == "process-flow-time":
            self.model.setObjective(gp.quicksum((self.z[i, m, t] - (self.z[i, m, t-1] if t > 0 else 0)) * (t + self.p[i][m] - self.ES[i]) for i in self.O for m in range(self.M[i]) for t in z_t[i]), GRB.MINIMIZE)

        # Constraints
        # Schedule each job exactly once
        self.model.addConstrs((gp.quicksum(self.z[i, m, self.LS[i]] for m in range(self.M[i])) == 1 for i in range(self.n)), name="schedule")
        self.model.addConstrs((gp.quicksum(self.z[i, m, self.T-1] for m in range(self.M[i])) == 1 for i in range(self.n)), name="schedule")

        # If job is started, at or before $t-1$ in mode $m$, it has also started before $t$ in mode $m$
        self.model.addConstrs((self.z[i, m, t-1] <= self.z[i, m, t] for i in range(self.n) for m in range(self.M[i]) for t in range(1, self.T)), name="started_same_mode")

        # Precedence relations between jobs (i,j)
        self.model.addConstrs((
            gp.quicksum((t + self.p[i][m]) * (self.z[i, m, t] - (self.z[i, m, t-1] if t-1>=0 else 0)) for t in range(self.T) for m in range(self.M[i])) <= 
            gp.quicksum(t * (self.z[j, m, t] - (self.z[j, m, t-1] if t-1>=0 else 0)) for t in range(self.T) for m in range(self.M[j]))
            for i,j in self.E), 
            name="precedence")

        # Resource availability
        self.model.addConstrs((gp.quicksum(self.r[i][m][k] * (self.z[i, m, t] - (self.z[i, m, max(t - self.p[i][m], 0)] if t-self.p[i][m]>=0 else 0)) for i in range(self.n) for m in range(self.M[i])) <= self.R[k] 
                     for t in range(self.T) for k in range(len(self.R))), name="resource")

        # Linked modes of jobs (i,j)
        self.model.addConstrs((self.z[i, m, self.T-1] == self.z[j, m, self.T-1] for i,j in self.L for m in range(self.M[i])), name="linked")
        
        # Earliest start times
        self.model.addConstrs((self.z[i, m, t] == 0 for i in range(self.n) for m in range(self.M[i]) for t in range(self.ES[i])), name="earliest_start_times")
        
        return self.model

    def visualize(self, filename: str) -> None:
        return None


class StepModelDisaggregated(GurobiModel):
    def initialize_model(self):
        # Initialize model
        self.model = gp.Model("step_disaggregated")
        if self.silent:
            self.model.setParam('OutputFlag', False)
        self.model.setParam('TimeLimit', self.timeout)

        # Step variables
        z_t = {
            i: range(self.ES[i], min(self.LS[i] + 1, self.T))
            for i in range(self.n)
        }
        step_sets = [(i, m, t) for i in range(self.n) for m in range(self.M[i]) for t in range(self.T)]
        self.z = self.model.addVars(step_sets, vtype=GRB.BINARY, name="step")

        # Objective
        if self.obj == "makespan":
            self.model.setObjective(gp.quicksum(t * (self.z[self.n-1, m, t] - self.z[self.n-1, m, t-1]) for t in range(1,self.T) for m in range(self.M[self.n-1])), GRB.MINIMIZE)
        elif self.obj == "flow-time":
            #model.setObjective(gp.quicksum((self.z[i, m, t] - self.z[i, m, t-1]) * (t + self.p[i][m] - self.ES[i]) if t > 0 else (self.z[i, m, t]) * (t + self.p[i][m] - self.ES[i]) for i in range(self.n) for m in range(self.M[i]) for t in z_t[i]), GRB.MINIMIZE)
            self.model.setObjective(gp.quicksum(self.p[i][m] + gp.quicksum(1 - (self.z[i, m, t]) for t in z_t[i]) for i in range(self.n) for m in range(self.M[i])), GRB.MINIMIZE)
        elif self.obj == "process-flow-time":
            self.model.setObjective(gp.quicksum((self.z[i, m, t] - (self.z[i, m, t-1] if t > 0 else 0)) * (t + self.p[i][m] - self.ES[i]) for i in self.O for m in range(self.M[i]) for t in z_t[i]), GRB.MINIMIZE)

        # Constraints
        # Schedule each job exactly once
        self.model.addConstrs((gp.quicksum(self.z[i, m, self.LS[i]] for m in range(self.M[i])) == 1 for i in range(self.n)), name="schedule")
        self.model.addConstrs((gp.quicksum(self.z[i, m, self.T-1] for m in range(self.M[i])) == 1 for i in range(self.n)), name="schedule")

        # If job is started, at or before $t-1$ in mode $m$, it has also started before $t$ in mode $m$
        self.model.addConstrs((self.z[i, m, t-1] <= self.z[i, m, t] for i in range(self.n) for m in range(self.M[i]) for t in range(1, self.T)), name="started_same_mode")

        # Precedence relations between jobs (i,j)
        self.model.addConstrs((
            gp.quicksum(self.z[i, m, max(t - self.p[i][m], 0)] if t-self.p[i][m]>=0 else 0 for m in range(self.M[i])) >= 
            gp.quicksum(self.z[j, m, t] for m in range(self.M[j])) for i,j in self.E for t in range(self.T)), 
            name="precedence_disaggregated")

        # Resource availability
        self.model.addConstrs((gp.quicksum(self.r[i][m][k] * (self.z[i, m, t] - (self.z[i, m, max(t - self.p[i][m], 0)] if t-self.p[i][m]>=0 else 0)) for i in range(self.n) for m in range(self.M[i])) <= self.R[k] 
                     for t in range(self.T) for k in range(len(self.R))), name="resource")

        # Linked modes of jobs (i,j)
        self.model.addConstrs((self.z[i, m, self.T-1] == self.z[j, m, self.T-1] for i,j in self.L for m in range(self.M[i])), name="linked")
        
        # Earliest start times
        self.model.addConstrs((self.z[i, m, t] == 0 for i in range(self.n) for m in range(self.M[i]) for t in range(self.ES[i])), name="earliest_start_times")
        
        return self.model

    def visualize(self, filename: str) -> None:
        return None


