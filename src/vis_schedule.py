import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from gurobipy import GRB
from instances.definitions import NetworkType


def visualize_pulse_model(
    model,
    n: int,
    T: int,
    M: int,
    R: list[int],
    p: list[list[int]],
    r: list[list[list[int]]],
    processes,
    divisor: int = 1,
    activity_names: list[str] | None = None,
    filename: str = "schedule",
) -> None:
    """
    Visualizes the solution of a solved pulse_model.

    model:          Solved Gurobi model returned by pulse_model
    n:              Number of activities
    T:              Number of time slots (pre-normalization)
    M:              Number of modes
    R:              List of resource capacities R[k]
    p:              Processing times p[i][m] (pre-normalization)
    r:              Resource requirements r[i][m][k]
    processes:      List of Process objects
    divisor:        Normalization divisor returned by pulse_model
    activity_names: Optional list of labels for each activity
    filename:       Output filename (saved under plots/)
    """
    if activity_names is None:
        activity_names = [f"A{i}" for i in range(n)]

    T_norm = T // divisor
    K = len(R)

    # ── Extract solution ──────────────────────────────────────────────────────
    # schedule[i] = (mode, t_start, t_end) in normalized time
    schedule: dict[int, tuple[int, int, int]] = {}
    vars_map = {v.VarName: v.X for v in model.getVars()}

    for i in range(n):
        for m in range(M):
            for t in range(T_norm):
                if vars_map.get(f"pulse[{i},{m},{t}]", 0) > 0.5:
                    duration = p[i][m] // divisor
                    t_end = min(t + duration, T_norm)
                    schedule[i] = (m, t, t_end)
                    break

    # ── Colour palette ────────────────────────────────────────────────────────
    # Activity 0 (dummy source) → gold
    # Activities 1 … n-2 grouped by process, sampled from tab20
    # Activity n-1 (dummy sink) → crimson
    n_middle = len(processes)
    group_cmap = plt.colormaps["tab20"]
    sample_points = np.linspace(0, 1, n_middle + 2, endpoint=False)[1:]

    colors = ["gold"]
    colors += [group_cmap(pt) for pt in sample_points[:n_middle]]
    colors.append("crimson")

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_plots = 1 + K
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(14, 3 + 2.5 * n_plots),
        gridspec_kw={"height_ratios": [max(n * 0.5, 3)] + [2] * K},
    )
    if n_plots == 1:
        axes = [axes]

    # ── Gantt chart ───────────────────────────────────────────────────────────
    ax_gantt = axes[0]

    current_color = 1
    y = 1
    yticks: list[int] = []
    ylabels: list[str] = []

    for i, process in enumerate(processes):
        for phase in range(len(process.phases)):
            if process.network_type == NetworkType.SINGLE and phase >= 1:
                continue
            if process.network_type == NetworkType.DOUBLE and phase >= 2:
                continue
            for _job in range(process.phases[phase].number_of_tasks):
                if y not in schedule:
                    y += 1
                    continue

                yticks.append(y)
                ylabels.append(activity_names[y])

                m, t_start, t_end = schedule[y]
                width = t_end - t_start

                bar = mpatches.FancyBboxPatch(
                    (t_start, y - 0.4),
                    max(width, 0),
                    0.8,
                    boxstyle="round,pad=0.03",
                    facecolor=colors[current_color],
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax_gantt.add_patch(bar)
                ax_gantt.text(
                    t_start + width / 2,
                    y,
                    f"m{i}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )
                y += 1

        current_color += 1

    ax_gantt.set_xlim(0, T_norm)
    ax_gantt.set_ylim(-0.7, n - 0.3)
    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(ylabels, fontsize=9)
    ax_gantt.set_xlabel("Time (normalized)", fontsize=10)
    ax_gantt.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_gantt.grid(axis="x", linestyle="--", alpha=0.4)
    ax_gantt.set_axisbelow(True)

    obj_label = (
        f"Makespan: {model.objVal * divisor:.0f}"
        if model.SolCount > 0 and model.objVal is not None
        else "No solution"
    )
    ax_gantt.set_title(
        f"Schedule – Gantt Chart    ({obj_label})",
        fontweight="bold",
        fontsize=13,
    )

    # ── Resource utilization per time slot ────────────────────────────────────
    for k in range(K):
        ax_res = axes[1 + k]
        usage = np.zeros(T_norm, dtype=float)

        for i, (m, t_start, t_end) in schedule.items():
            for t in range(t_start, min(t_end, T_norm)):
                usage[t] += r[i][m][k]

        ax_res.bar(
            range(T_norm),
            usage,
            color="steelblue",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            width=1.0,
            align="edge",
        )
        ax_res.axhline(
            R[k],
            color="crimson",
            linewidth=1.5,
            linestyle="--",
            label=f"Capacity = {R[k]}",
        )
        ax_res.set_xlim(0, T_norm)
        ax_res.set_ylim(0, R[k] * 1.25)
        ax_res.set_title(
            f"Resource {k} utilization", fontweight="bold", fontsize=11
        )
        ax_res.set_xlabel("Time (normalized)", fontsize=10)
        ax_res.set_ylabel("Usage", fontsize=10)
        ax_res.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_res.legend(fontsize=9)
        ax_res.grid(axis="y", linestyle="--", alpha=0.4)
        ax_res.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    plt.savefig(f"plots/{filename}.png")
    plt.close()


def visualize_continuous_model(
    model,
    n: int,
    T: int,
    M: int,
    R: list[int],
    p: list[list[int]],
    r: list[list[list[int]]],
    processes,
    divisor: int = 1,
    activity_names: list[str] | None = None,
    filename: str = "schedule_continuous",
) -> None:
    """
    Visualizes the solution of a solved continuous_model.

    model:          Solved Gurobi model returned by continuous_model
    n:              Number of activities
    T:              Number of time slots (pre-normalization)
    M:              Number of modes
    R:              List of resource capacities R[k]
    p:              Processing times p[i][m] (pre-normalization)
    r:              Resource requirements r[i][m][k]
    processes:      List of Process objects
    divisor:        Normalization divisor returned by continuous_model
    activity_names: Optional list of labels for each activity
    filename:       Output filename (saved under plots/)
    """
    if activity_names is None:
        activity_names = [f"A{i}" for i in range(n)]

    T_norm = T // divisor
    K = len(R)

    # ── Extract solution ──────────────────────────────────────────────────────
    # schedule[i] = (mode, t_start, t_end) in normalized time
    # continuous model uses S[i] for start times and x[i,m] for mode selection
    schedule: dict[int, tuple[int, int, int]] = {}
    vars_map = {v.VarName: v.X for v in model.getVars()}

    for i in range(n):
        t_start = vars_map.get(f"activity[{i}]")
        if t_start is None:
            continue
        t_start = int(round(t_start))
        for m in range(M):
            if vars_map.get(f"mode[{i},{m}]", 0) > 0.5:
                duration = p[i][m] // divisor
                t_end = min(t_start + duration, T_norm)
                schedule[i] = (m, t_start, t_end)
                break

    # ── Colour palette ────────────────────────────────────────────────────────
    n_middle = len(processes)
    group_cmap = plt.colormaps["tab20"]
    sample_points = np.linspace(0, 1, n_middle + 2, endpoint=False)[1:]

    colors = ["gold"]
    colors += [group_cmap(pt) for pt in sample_points[:n_middle]]
    colors.append("crimson")

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_plots = 1 + K
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(14, 3 + 2.5 * n_plots),
        gridspec_kw={"height_ratios": [max(n * 0.5, 3)] + [2] * K},
    )
    if n_plots == 1:
        axes = [axes]

    # ── Gantt chart ───────────────────────────────────────────────────────────
    ax_gantt = axes[0]

    current_color = 1
    y = 1
    yticks: list[int] = []
    ylabels: list[str] = []

    for i, process in enumerate(processes):
        for phase in range(len(process.phases)):
            if process.network_type == NetworkType.SINGLE and phase >= 1:
                continue
            if process.network_type == NetworkType.DOUBLE and phase >= 2:
                continue
            for _job in range(process.phases[phase].number_of_tasks):
                if y not in schedule:
                    y += 1
                    continue

                yticks.append(y)
                ylabels.append(activity_names[y])

                m, t_start, t_end = schedule[y]
                width = t_end - t_start

                bar = mpatches.FancyBboxPatch(
                    (t_start, y - 0.4),
                    max(width, 0),
                    0.8,
                    boxstyle="round,pad=0.03",
                    facecolor=colors[current_color],
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax_gantt.add_patch(bar)
                ax_gantt.text(
                    t_start + width / 2,
                    y,
                    f"m{m}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )
                y += 1

        current_color += 1

    ax_gantt.set_xlim(0, T_norm)
    ax_gantt.set_ylim(-0.7, n - 0.3)
    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(ylabels, fontsize=9)
    ax_gantt.set_xlabel("Time (normalized)", fontsize=10)
    ax_gantt.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_gantt.grid(axis="x", linestyle="--", alpha=0.4)
    ax_gantt.set_axisbelow(True)

    obj_label = (
        f"Objective: {model.objVal * divisor:.0f}"
        if model.SolCount > 0 and model.objVal is not None
        else "No solution"
    )
    ax_gantt.set_title(
        f"Schedule – Gantt Chart    ({obj_label})",
        fontweight="bold",
        fontsize=13,
    )

    # ── Resource utilization ──────────────────────────────────────────────────
    # Unlike pulse, S[i] is continuous so we compute utilization over a fine
    # integer grid: activity i uses r[i][m][k] during [t_start, t_end)
    for k in range(K):
        ax_res = axes[1 + k]
        usage = np.zeros(T_norm, dtype=float)

        for i, (m, t_start, t_end) in schedule.items():
            for t in range(t_start, min(t_end, T_norm)):
                usage[t] += r[i][m][k]

        ax_res.bar(
            range(T_norm),
            usage,
            color="steelblue",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            width=1.0,
            align="edge",
        )
        ax_res.axhline(
            R[k],
            color="crimson",
            linewidth=1.5,
            linestyle="--",
            label=f"Capacity = {R[k]}",
        )
        ax_res.set_xlim(0, T_norm)
        ax_res.set_ylim(0, R[k] * 1.25)
        ax_res.set_title(
            f"Resource {k} utilization", fontweight="bold", fontsize=11
        )
        ax_res.set_xlabel("Time (normalized)", fontsize=10)
        ax_res.set_ylabel("Usage", fontsize=10)
        ax_res.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_res.legend(fontsize=9)
        ax_res.grid(axis="y", linestyle="--", alpha=0.4)
        ax_res.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    plt.savefig(f"plots/{filename}.png")
    plt.close()
