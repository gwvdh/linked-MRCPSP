import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from .utils import normalize
from .model import GurobiModel
from .vis_schedule import visualize_pulse_model


class PulseModel(GurobiModel):
    def initialize_model(self):
        # Initialize model
        self.model = gp.Model("pulse")
        if self.silent:
            self.model.setParam('OutputFlag', False)
        self.model.setParam('TimeLimit', self.timeout)

        # Pulse variables
        x_t = {
            i: range(self.ES[i], min(self.LS[i] + 1, self.T))
            for i in range(self.n)
        }
        pulse_sets = [(i, m, t) for i in range(self.n) for m in range(self.M[i]) for t in x_t[i]]
        self.x = self.model.addVars(pulse_sets, vtype=GRB.BINARY, name="pulse")

        # Objective
        if self.obj == "makespan":
            self.model.setObjective(gp.quicksum(t * self.x[self.n-1, m, t] for t in x_t[self.n-1] for m in range(self.M[self.n-1])), GRB.MINIMIZE)
        elif self.obj == "flow-time":
            self.model.setObjective(gp.quicksum(self.x[i, m, t] * (t + self.p[i][m] - self.ES[i]) for i in range(self.n) for m in range(self.M[i]) for t in x_t[i]), GRB.MINIMIZE)
        elif self.obj == "process-flow-time":
            self.model.setObjective(gp.quicksum(self.x[i, m, t] * (t + self.p[i][m] - self.ES[i]) for i in self.O for m in range(self.M[i]) for t in x_t[i]), GRB.MINIMIZE)

        # Constraints
        # Schedule job exactly once
        self.model.addConstrs((self.x.sum(i, "*", "*") == 1 for i in range(self.n)), name="schedule")

        # Precedence relations between jobs (i,j)
        self.model.addConstrs((
            gp.quicksum((t + self.p[i][m]) * self.x[i, m, t] for m in range(self.M[i]) for t in x_t[i]) <= 
            gp.quicksum(t * self.x[j, m, t] for m in range(self.M[j]) for t in x_t[j])
            for i,j in self.E), 
            name="precedence")

        # Resource availability
        self.model.addConstrs((
            gp.quicksum(self.r[i][m][k] * self.x[i, m, tau] 
                for i in range(self.n) 
                for m in range(self.M[i]) 
                for tau in range(
                    max(t-self.p[i][m]+1, self.ES[i]), 
                    min(t, self.LS[i])+1
                )
            ) <= self.R[k] 
            for t in range(self.T) for 
            k in range(len(self.R))
        ), name="resource")

        # Linked modes of jobs (i,j)
        self.model.addConstrs((gp.quicksum(self.x[i, m, t] for t in x_t[i]) <= gp.quicksum(self.x[j, m, t] for t in x_t[j]) 
                              for i,j in self.L for m in range(self.M[i])), name="linked")
        
        # Zero time slots 
        #self.model.addConstrs((gp.quicksum(self.x[i, m, t] for t in x_t[i]) == 0 for i in range(self.n) for m in range(self.M[i])), name="zero_time_slots")
        
        return self.model

    def visualize(self, filename: str) -> None:
        visualize_pulse_model(self.model, self.n, self.T, self.M, self.R, self.p, self.r, self.processes, self.divisor, activity_names=None, filename=filename)


class PulseModelDisaggregated(PulseModel):
    def initialize_model(self):
        # Initialize model
        self.model = gp.Model("pulse_disaggregated")
        if self.silent:
            self.model.setParam('OutputFlag', False)
        self.model.setParam('TimeLimit', self.timeout)

        # Pulse variables
        x_t = {
            i: range(self.ES[i], min(self.LS[i] + 1, self.T))
            for i in range(self.n)
        }
        pulse_sets = [(i, m, t) for i in range(self.n) for m in range(self.M[i]) for t in x_t[i]]
        self.x = self.model.addVars(pulse_sets, vtype=GRB.BINARY, name="pulse")

        # Objective
        if self.obj == "makespan":
            self.model.setObjective(gp.quicksum(t * self.x[self.n-1, m, t] for t in range(self.T) for m in range(self.M[self.n-1])), GRB.MINIMIZE)
        elif self.obj == "flow-time":
            self.model.setObjective(gp.quicksum(self.x[i, m, t] * (t + self.p[i][m] - self.ES[i]) for i in range(self.n) for m in range(self.M[i]) for t in x_t[i]), GRB.MINIMIZE)
        elif self.obj == "process-flow-time":
            self.model.setObjective(gp.quicksum(self.x[i, m, t] * (t + self.p[i][m] - self.ES[i]) for i in self.O for m in range(self.M[i]) for t in x_t[i]), GRB.MINIMIZE)

        # Constraints
        # Schedule job exactly once
        self.model.addConstrs((self.x.sum(i, "*", "*") == 1 for i in range(self.n)), name="schedule")

        # Precedence relations between jobs (i,j)
        self.model.addConstrs((
            gp.quicksum(self.x[i, m, tau] for m in range(self.M[i]) for tau in range(
                self.ES[i],
                min(self.LS[i], t-self.p[i][m])+1
            )) >=
            gp.quicksum(self.x[j, m, tau] for m in range(self.M[j]) for tau in range(self.ES[j], min(self.LS[j], t)+1))
            for i,j in self.E for t in range(self.T)), 
            name="precedence_disaggregated")

        # Resource availability
        self.model.addConstrs((
            gp.quicksum(self.r[i][m][k] * self.x[i, m, tau] 
                for i in range(self.n) 
                for m in range(self.M[i]) 
                for tau in range(
                    max(t-self.p[i][m]+1, self.ES[i]), 
                    min(t+1, self.LS[i]+1)
                )
            ) <= self.R[k] 
            for t in range(self.T) for 
            k in range(len(self.R))
        ), name="resource")

        # Linked modes of jobs (i,j)
        self.model.addConstrs((gp.quicksum(self.x[i, m, t] for t in x_t[i]) <= gp.quicksum(self.x[j, m, t] for t in x_t[j]) 
                              for i,j in self.L for m in range(self.M[i])), name="linked")
        
        # Zero time slots 
        #self.model.addConstrs((self.x[i, m, t] == 0 for i in range(self.n) for m in range(self.M[i]) for t in range(self.ES[i], self.LS[i]+1)), name="zero_time_slots")
        
        return self.model

    def visualize(self, filename: str) -> None:
        visualize_pulse_model(self.model, self.n, self.T, self.M, self.R, self.p, self.r, self.processes, self.divisor, activity_names=None, filename=filename)


