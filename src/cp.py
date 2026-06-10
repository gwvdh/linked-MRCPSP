from ortools.sat.python import cp_model
import json
import time
import collections

if __name__ == "__main__":
    from utils import normalize
    import pickle
else:
    from .utils import normalize
    from .model import CP_SATModel
    from .vis_schedule import visualize_cp_model


class ConstraintProgrammingModel(CP_SATModel):
    def initialize_model(self):
        # Initialize model
        self.model = cp_model.CpModel()

        task_type = collections.namedtuple("task_type", "start end interval is_present")
        assigned_task_type = collections.namedtuple(
            "assigned_task_type", "start job index duration"
        )

        all_tasks = {}
        resource_to_intervals = collections.defaultdict(list)

        # activities
        for i in range(self.n):
            present_vars = []
            for task_id, task in enumerate(range(self.M[i])):
                if sum(self.r[i][task]) != 1:
                    resource = 0
                else:
                    resource = self.r[i][task].index(1)
                duration = self.p[i][task]
                suffix = f"_{i}_{task_id}"
                start_var = self.model.new_int_var(self.ES[i], self.LS[i], "start" + suffix)
                end_var = self.model.new_int_var(self.ES[i], self.LS[i]+max(self.p[i]), "end" + suffix)
                is_present_var = self.model.new_bool_var(f"is_present{suffix}")
                present_vars.append(is_present_var)
                interval_var = self.model.new_optional_interval_var(start=start_var, size=duration, end=end_var, is_present=is_present_var, name="interval" + suffix)
                all_tasks[i, task_id] = task_type(
                    start=start_var,
                    end=end_var,
                    interval=interval_var,
                    is_present=is_present_var,
                )
                resource_to_intervals[resource].append(interval_var)
            # Present in exactly one mode
            self.model.add(sum(present_vars) == 1)

        # Resource overlap constraints
        for resource, intervals in resource_to_intervals.items():
            self.model.add_cumulative(
                intervals=intervals,
                demands=[1] * len(intervals),
                capacity=self.R[resource],
            )

        # Precedence constraints
        for i, j in self.E:
            for task_id in range(self.M[i]):
                for task_id2 in range(self.M[j]):
                    self.model.add(
                        all_tasks[j, task_id2].start >= all_tasks[i, task_id].end
                    ).only_enforce_if([all_tasks[i, task_id].is_present, all_tasks[j, task_id2].is_present])

        # Linked modes
        for i, j in self.L:
            for task_id in range(self.M[i]):
                self.model.add(all_tasks[i, task_id].is_present == all_tasks[j, task_id].is_present)

        completion_vars = []
        for i in range(self.n):
            completion = self.model.new_int_var(0, self.T, f"completion_{i}")
            completion_vars.append(completion)
            for task_id in range(self.M[i]):
                task = all_tasks[i, task_id]
                self.model.add(completion == task.end).only_enforce_if(task.is_present)

        if self.obj == "makespan":
            obj_var = self.model.new_int_var(0, self.T, "makespan")
            self.model.add_max_equality(obj_var, completion_vars)
            self.model.minimize(obj_var)
        elif self.obj == "flow-time":
            obj_var = self.model.new_int_var(0, self.T*self.n, "flow_time")
            self.model.add(obj_var == sum(completion_vars[i] - self.ES[i] for i in range(self.n)))
            self.model.minimize(obj_var)

        return self.model

    def visualize(self, filename: str) -> None:
        visualize_cp_model(self.model, self.solver, self.n, self.T, self.M, self.R, self.p, self.r, self.processes, self.divisor, activity_names=None, filename=filename)


