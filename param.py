from __future__ import annotations

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from typing import Any, Dict

MODELS = ["PDT", "PDDT", "SDT", "SDDT", "OODDT", "OOPDT", "OOPDDT", "MSEQCT"]
SCARCITIES = [round(s * 0.1, 1) for s in range(11)]

if __name__ == "__main__":
    from database import Database
    from instances.generator import simulate_processes, _collect_resource_indices
else:
    from .database import Database
    from .instances.generator import simulate_processes, _collect_resource_indices

_FEATURES = [
    #"number_of_processes", 
    #"min_resource_ratio",
    #"resource_ratio_center", 
    #"resource_ratio_spread", 
    "resource_variance", 
    "schedule_density",
    "duration_flexibility",
    "VP_mean",
    #"scarcity",
]


def demand_variance_over_time(processes, max_phases: int, n_simulations: int = 20, normalise: bool = True) -> float:
    """
    compute the variance of the demand over time for the processes, averaged over n_simulations
    """
    all_resource_indices = _collect_resource_indices(processes)
    T_max = max(int(p.start_time + p.max_processing_time()) for p in processes)
    stds: Dict[int, list] = defaultdict(list)
    for _ in range(n_simulations):
        timelines = simulate_processes(processes, max_phases=max_phases)
        for resource_idx in all_resource_indices:
            profile = np.zeros(T_max + 1)
            for phase_timeline in timelines:
                for t, res_counts in phase_timeline.items():
                    if t <= T_max:
                        profile[t] += res_counts.get(resource_idx, 0)
            mean = np.mean(profile)
            std = np.std(profile)
            stds[resource_idx].append(std/mean if (normalise and np.isfinite(mean) and mean > 0) else std)
    return float(np.nanmean([np.nanmean(stds[r]) for r in all_resource_indices]))


def _read_sol(db: Database, iid: int, model: str, scarcity: float):
    """Return (objective, runtime) or (None, None) if unavailable."""
    sol = db.get_solution(iid, model, scarcity)
    if sol is None:
        return None, None
    try:
        with open(sol["sol_file"], "r") as f:
            info = json.load(f)["SolutionInfo"]
    except (FileNotFoundError, KeyError):
        return None, None
    if info["SolCount"] == 0:
        return None, None
    return info["ObjVal"] * sol["divisor"], info["Runtime"]


def plot_objective_heatmap(
        db: Database, 
        instance_ids: list[int] | None = None,
        filename: str = "objective_heatmap", 
) -> None:
    """
    Generate a heatmap of objective values and running times for all instances.
    """
    if instance_ids is None:
        instance_ids = [r["id"] for r in db.get_instances()]
    for iid in instance_ids:
        obj = np.full((len(MODELS), len(SCARCITIES)), np.nan)
        rt  = np.full((len(MODELS), len(SCARCITIES)), np.nan)
        for i, m in enumerate(MODELS):
            for j, s in enumerate(SCARCITIES):
                obj[i, j], rt[i, j] = _read_sol(db, iid, m, s)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, data, title, cmap in [
            (axes[0], obj, "Objective Value", "YlOrRd"),
            (axes[1], rt,  "Runtime (s)",     "YlOrBr"),
        ]:
            im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(SCARCITIES)))
            ax.set_xticklabels([f"{s:.1f}" for s in SCARCITIES])
            ax.set_yticks(range(len(MODELS)))
            ax.set_yticklabels(MODELS)
            ax.set_xlabel("Scarcity")
            ax.set_title(f"Instance {iid} – {title}", fontweight="bold")
            for (r, c), v in np.ndenumerate(data):
                ax.text(c, r, "–" if np.isnan(v) else f"{v:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="gray" if np.isnan(v) else "black")

        plt.tight_layout()
        plt.savefig(f"plots/{filename}_{iid}_heatmap.png", dpi=150)
        plt.close()

def get_resource_variance(db: Database, instance_ids: list[int] | None = None) -> dict[int, float]:
    """
    Compute the variance of the demand over time for the processes, averaged over n_simulations
    """
    resource_variance = {}
    for iid in instance_ids:
        row = db.get_instance(iid)
        with open(row["processes_file"], "rb") as f:
            processes, _ = pickle.load(f)

            demand_variance = demand_variance_over_time(processes, max_phases=3, n_simulations=100, normalise=True)
            print(f"Resource variance (iid={iid}): {demand_variance}")
            resource_variance[iid] = demand_variance
    return resource_variance


def get_schedule_density(db: Database, instance_ids: list[int] | None = None) -> dict[int, float]:
    """
    Compute the schedule density for the processes. 
    """
    schedule_density = {}
    for iid in instance_ids:
        row = db.get_instance(iid)
        with open(row["processes_file"], "rb") as f:
            processes, _ = pickle.load(f)

            total_work = sum(p.max_processing_time() for p in processes)
            max_T = max(int(p.start_time + p.max_processing_time()) for p in processes)
            schedule_density[iid] = total_work / (row["number_of_processes"] * max_T)
        print(f"Schedule density (iid={iid}): {schedule_density[iid]}")
    return schedule_density


def get_duration_flexibility(db: Database, instance_ids: list[int] | None = None) -> dict[int, float]:
    """
    Compute the duration flexibility for the processes. 
    """
    duration_flexibility = {}
    for iid in instance_ids:
        row = db.get_instance(iid)
        with open(row["processes_file"], "rb") as f:
            processes, _ = pickle.load(f)

            tasks = []
            for p in processes:
                for phase_tasks in p.tasks:
                    for task in phase_tasks:
                        task_std = np.std(task.duration)
                        task_mean = np.mean(task.duration)
                        if task_mean > 0.0:
                            tasks.append(task_std / task_mean)
            duration_flexibility[iid] = np.mean(tasks)
        print(f"Duration flexibility (iid={iid}): {duration_flexibility[iid]}")
    return duration_flexibility


def get_VP_mean(db: Database, instance_ids: list[int] | None = None) -> dict[int, float]:
    """
    Compute the mean of the number of non-connected activities. 
    """
    VP_mean = {}
    for iid in instance_ids:
        row = db.get_instance(iid)
        instance_file = db.get_solution(iid, "PDT", 0.0)["instance_file"]
        # Load solution json file
        with open(instance_file, "r") as f:
            instance = json.load(f)
            VP_size = len(instance["VP"])
            n = instance["n"]
            VP_mean[iid] = VP_size / (n*n)
        print(f"VP mean (iid={iid}): {VP_mean[iid]}")
    return VP_mean

def plot_instance_space_scarcities(
    db: Database,
    instance_ids: list[int] | None = None,
    filename: str = "instance_space_scarcities",
) -> None:
    """
    Generate instance space analysis plots and save to plots.
    """
    os.makedirs("plots", exist_ok=True)
    if instance_ids is None:
        instance_ids = [r["id"] for r in db.get_instances()]

    plot_objective_heatmap(db, filename=f"instance_space_scarcities", instance_ids=instance_ids)
    if len(instance_ids) < 2:
        return

    # Additional features
    resource_variance = get_resource_variance(db, instance_ids)
    schedule_density = get_schedule_density(db, instance_ids)
    duration_flexibility = get_duration_flexibility(db, instance_ids)
    VP_mean = get_VP_mean(db, instance_ids)

    rows = []
    for iid in instance_ids:
        for scarcity in SCARCITIES:
            r: Dict[str, Any] = {} # feature: value
            for f in _FEATURES:
                if f == "resource_variance":
                    r[f] = resource_variance[iid]
                elif f == "schedule_density":
                    r[f] = schedule_density[iid]
                elif f == "duration_flexibility":
                    r[f] = duration_flexibility[iid]
                elif f == "VP_mean":
                    r[f] = VP_mean[iid]
                elif f == "scarcity":
                    r[f] = scarcity
                else:
                    r[f] = db.get_instance(iid)[f]
            rows.append(r)
    X = np.array([[r[f] for f in _FEATURES] for r in rows], dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    evals, evecs = np.linalg.eigh(np.cov(X.T))
    order = evals.argsort()[::-1]
    proj  = X @ evecs[:, order[:2]]
    var   = evals[order] / evals.sum()

    # best solver = lowest mean objective across scarcities (ignoring infeasible)
    best = []
    for iid in instance_ids:
        for s in SCARCITIES:
            scores = {}
            for m in MODELS:
                obj, t = _read_sol(db, iid, m, s)
                if t is None: 
                    t = db.get_instance(iid)["timeout"]
                scores[m] = t
            best.append(min(scores, key=scores.get) if scores else "N/A")

    solvers = sorted(set(best))
    cmap    = plt.colormaps["tab10"]
    colors  = {s: cmap(i / max(len(solvers) - 1, 1)) for i, s in enumerate(solvers)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for (iid, scarcity), (x, y), solver in zip(
        ((iid, s) for iid in instance_ids for s in SCARCITIES), 
        proj, best):
        #print(
        #    db.get_solution(iid, solver, scarcity)["solved"],
        #    db.get_solution(iid, solver, scarcity)["status"],
        #    db.get_solution(iid, solver, scarcity)["objective"],
        #    db.get_solution(iid, solver, scarcity)["objective_val"],
        #    sep="\t"
        #    )
        if db.get_solution(iid, solver, scarcity)["objective_val"] is None: continue
        ax.scatter(x, y, color=colors[solver], s=80, zorder=3)
        ax.annotate(f"{iid} ({scarcity})", (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[s], markersize=9, label=s)
            for s in solvers
        ],
        title="Best solver", fontsize=9,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title("Instance Space – PCA (colored by best solver)", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}_pca.png", dpi=150)
    plt.close()


def plot_instance_space(
    db: Database,
    instance_ids: list[int] | None = None,
    filename: str = "instance_space",
) -> None:
    """
    Generate instance space analysis plots and save to plots/.

    Per instance:
        - Objective value heatmap  (solver × scarcity)
        - Runtime heatmap          (solver × scarcity)

    Across instances (requires ≥ 2):
        - PCA scatter of instance features, colored by best-performing solver
    """
    os.makedirs("plots", exist_ok=True)
    if instance_ids is None:
        instance_ids = [r["id"] for r in db.get_instances()]

    plot_objective_heatmap(db, filename=f"instance_space", instance_ids=instance_ids)
    if len(instance_ids) < 2:
        return

    # Additional features
    resource_variance = get_resource_variance(db, instance_ids)
    schedule_density = get_schedule_density(db, instance_ids)
    duration_flexibility = get_duration_flexibility(db, instance_ids)
    VP_mean = get_VP_mean(db, instance_ids)

    rows = []
    for iid in instance_ids:
        r: Dict[str, Any] = {} # feature: value
        for f in _FEATURES:
            if f == "resource_variance":
                r[f] = resource_variance[iid]
            elif f == "schedule_density":
                r[f] = schedule_density[iid]
            elif f == "duration_flexibility":
                r[f] = duration_flexibility[iid]
            elif f == "VP_mean":
                r[f] = VP_mean[iid]
            else:
                r[f] = db.get_instance(iid)[f]
        rows.append(r)

    X = np.array([[r[f] for f in _FEATURES] for r in rows], dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    evals, evecs = np.linalg.eigh(np.cov(X.T))
    order = evals.argsort()[::-1]
    proj  = X @ evecs[:, order[:2]]
    var   = evals[order] / evals.sum()

    # best solver = lowest mean objective across scarcities (ignoring infeasible)
    best = []
    for iid in instance_ids:
        scores = {}
        for m in MODELS:
            vals = []
            for s in SCARCITIES:
                obj, t = _read_sol(db, iid, m, s)
                if t is None: 
                    t = db.get_instance(iid)["timeout"]
                vals.append(t)
            if vals:
                scores[m] = np.mean(vals)
        best.append(min(scores, key=scores.get) if scores else "N/A")

    solvers = sorted(set(best))
    cmap    = plt.colormaps["tab10"]
    colors  = {s: cmap(i / max(len(solvers) - 1, 1)) for i, s in enumerate(solvers)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for iid, (x, y), solver in zip(instance_ids, proj, best):
        ax.scatter(x, y, color=colors[solver], s=80, zorder=3)
        ax.annotate(str(iid), (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[s], markersize=9, label=s)
            for s in solvers
        ],
        title="Best solver", fontsize=9,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title("Instance Space – PCA (colored by best solver)", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}_pca.png", dpi=150)
    plt.close()


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Instance space analysis"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    instance_space = sub.add_parser("instance-space", help="Plot instance space")
    instance_space.add_argument(
        "--instance-ids", nargs="+", default=None, dest="instance_ids",
        metavar="IID", help="Instance IDs to plot (default: all)",
    )
    instance_space.add_argument(
        "--filename", default="instance_space", help="Filename for plots"
    )
    instance_space.set_defaults(func=plot_instance_space)

    instance_space_scarcities = sub.add_parser(
        "instance-space-scarcities", help="Plot instance space analysis"
    )
    instance_space_scarcities.add_argument(
        "--instance-ids", nargs="+", default=None, dest="instance_ids",
        metavar="IID", help="Instance IDs to plot (default: all)",
    )
    instance_space_scarcities.add_argument(
        "--filename", default="instance_space_scarcities", help="Filename for plots"
    )
    instance_space_scarcities.set_defaults(func=plot_instance_space_scarcities)
    return parser

if __name__ == "__main__":
    db = Database("database.db")
    args = arg_parser().parse_args()
    default_instance_ids = [1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19]
    args.func(db, instance_ids=args.instance_ids or default_instance_ids, filename=args.filename)
