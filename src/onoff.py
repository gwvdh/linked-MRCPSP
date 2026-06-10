import gurobipy as gp
from gurobipy import GRB
import json
from math import gcd
import time
from .utils import normalize
from .model import GurobiModel
from .vis_schedule import visualize_onoff_model


class OnoffModel(GurobiModel):
    def initialize_model(self):
        # Initialize model
        self.model = gp.Model("onoff")
        if self.silent:
            self.model.setParam('OutputFlag', False)
        self.model.setParam('TimeLimit', self.timeout)

        # Step variables
        step_sets = [(i, m, t) for i in range(self.n) for m in range(self.M[i]) for t in range(self.T)]
        self.y = self.model.addVars(step_sets, vtype=GRB.BINARY, name="onoff")

        # Objective
        if self.obj == "makespan":
            self.model.setObjective(gp.quicksum(t * self.y[self.n-1, m, t] for t in range(1,self.T) for m in range(self.M[self.n-1])), GRB.MINIMIZE)
        elif self.obj == "flow-time":
            self.model.setObjective(gp.quicksum(
                gp.quicksum(gp.quicksum(self.y[i, m, t] * t for t in range(self.T))/max(self.p[i][m], 1) + gp.quicksum(self.y[i, m, t] for t in range(self.T))/max(self.p[i][m], 1) * ((self.p[i][m]/2.0) + (0.5 if self.p[i][m] > 0 else 0)) for m in range(self.M[i]))
            - self.ES[i] for i in range(self.n)), GRB.MINIMIZE)
        elif self.obj == "process-flow-time":
            # Rounding the addition down for odd length, rounding down the timeslot for even length
            self.model.setObjective(gp.quicksum(
                gp.quicksum(gp.quicksum(self.y[i, m, t] * t for t in range(self.T))/max(self.p[i][m], 1) + gp.quicksum(self.y[i, m, t] for t in range(self.T))/max(self.p[i][m], 1) * ((self.p[i][m]/2.0) + (0.5 if self.p[i][m] > 0 else 0)) for m in range(self.M[i]))
            - self.ES[i] for i in self.O), GRB.MINIMIZE)

        # Constraints
        # Schedule each job exactly once (schedule dummy separately, since p[n-1][m] = 0)
        self.model.addConstrs((gp.quicksum(self.y[i, m, t]/max(self.p[i][m], 1) for m in range(self.M[i]) for t in range(self.T) ) == 1 for i in range(self.n)), name="schedule")

        # If job is started, start it for exactly p[i][m] consecutive time slots in mode m
        self.model.addConstrs((self.p[i][m] * (self.y[i, m, t] - self.y[i, m, t+1]) - gp.quicksum(self.y[i, m, tau] for tau in range(max(t - self.p[i][m]+1, 0), t)) <= 1 for i in range(self.n) for m in range(self.M[i]) for t in range(self.T-1)), name="started_same_mode")

        # Precedence relations between jobs (i,j)
        self.model.addConstrs((gp.quicksum(self.y[i, m, tau]/max(self.p[i][m], 1) for m in range(self.M[i]) for tau in range(t + 1 - min(self.p[i][m], 1))) >= gp.quicksum(self.y[j, m, t] for m in range(self.M[j])) for i,j in self.E for t in range(self.T)), name="precedence")
        #model.addConstrs((gp.quicksum(self.y[i, m, tau]/max(self.p[i][m], 1) for m in range(self.M) for tau in range(t) if self.p[i][m] > 0) >= gp.quicksum(self.y[j, m, t] for m in range(self.M)) for i,j in self.E for t in range(self.T)), name="precedence")

        # Resource availability
        self.model.addConstrs((gp.quicksum(self.r[i][m][k] * self.y[i, m, t] for i in range(self.n) for m in range(self.M[i]) if self.p[i][m] > 0) <= self.R[k] for t in range(self.T) for k in range(len(self.R))), name="resource")

        # Linked modes of jobs (i,j)
        self.model.addConstrs((gp.quicksum(self.y[i, m, t] for t in range(self.T))/max(self.p[i][m],1) == gp.quicksum(self.y[j, m, t] for t in range(self.T))/max(self.p[j][m],1) for i,j in self.L for m in range(self.M[i])), name="linked")
        
        # Zero time slots 
        self.model.addConstrs((self.y[i, m, t] == 0 for i in range(self.n) for m in range(self.M[i]) for t in range(self.ES[i])), name="zero_time_slots")
        self.model.addConstrs((self.y[i, m, t] == 0 for i in range(self.n) for m in range(self.M[i]) for t in range(self.LS[i]+self.p[i][m]+1, self.T)), name="zero_time_slots")
        
        return self.model

    def visualize(self, filename: str) -> None:
        visualize_onoff_model(self.model, self.n, self.T, self.M, self.R, self.p, self.r, self.processes, self.divisor, activity_names=None, filename=filename)


