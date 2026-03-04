import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from gurobipy import GRB


def visualize_pulse_model(model, n, T, M, R, p, r, divisor=1, activity_names=None, filename="schedule"):
    """
    Visualizes the solution of a solved pulse_model.

    model:          Solved Gurobi model returned by pulse_model
    n:              Number of activities
    T:              Number of time slots (pre-normalization)
    M:              Number of modes
    R:              List of resource capacities R[k]
    p:              Processing times p[i][m] (pre-normalization)
    r:              Resource requirements r[i][m][k]
    divisor:        Normalization divisor returned by pulse_model
    activity_names: Optional list of labels for each activity
    """
    #if model.status != GRB.OPTIMAL and model.status != GRB.SUBOPTIMAL:
    #    print("Model has no feasible solution to visualize.")
    #    return

    if activity_names is None:
        activity_names = [f"A{i}" for i in range(n)]

    T_norm = T // divisor
    K = len(R)

    # ── Extract solution ──────────────────────────────────────────────────────
    schedule = {}  # i -> (m, t_start, t_end) in normalized time
    vars_map = {v.varName: v.x for v in model.getVars()}

    for i in range(n):
        for m in range(M):
            for t in range(T_norm):
                key = f"pulse[{i},{m},{t}]"
                if vars_map.get(key, 0) > 0.5:
                    duration = p[i][m] // divisor
                    schedule[i] = (m, t, t + duration)
                    break

    # ── Colour palette ────────────────────────────────────────────────────────
    # Activity 0 (dummy source) and n-1 (dummy sink) each get a distinct colour.
    # Activities 1 to n-2 are grouped in sets of 9 sharing the same colour.
    n_groups = -((-(n - 2)) // 9)  # ceil((n-2) / 9)
    group_cmap = plt.colormaps["prism"]
    group_colors = [group_cmap(g / n_groups) for g in range(n_groups)]

    colors = ["gold"]  # activity 0 (dummy source)
    colors += [group_colors[(i - 1) // 9] for i in range(1, n - 1)]
    colors.append("crimson")  # activity n-1 (dummy sink)

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
    ax_gantt.set_title("Schedule – Gantt Chart", fontweight="bold", fontsize=13)

    yticks, ylabels = [], []
    for i, (name, color) in enumerate(zip(activity_names, colors)):
        y = n - 1 - i  # top-to-bottom order
        yticks.append(y)
        ylabels.append(name)

        if i not in schedule:
            continue

        m, t_start, t_end = schedule[i]
        width = t_end - t_start

        bar = mpatches.FancyBboxPatch(
            (t_start, y - 0.4),
            width,
            0.8,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_gantt.add_patch(bar)
        ax_gantt.text(
            t_start + width / 2,
            y,
            f"m{(i - 1) // 9}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    ax_gantt.set_xlim(0, T_norm)
    ax_gantt.set_ylim(-0.7, n - 0.3)
    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(ylabels, fontsize=9)
    ax_gantt.set_xlabel("Time (normalized)", fontsize=10)
    ax_gantt.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_gantt.grid(axis="x", linestyle="--", alpha=0.4)
    ax_gantt.set_axisbelow(True)

    makespan_label = (
        f"Makespan: {model.objVal * divisor:.0f}" if model.objVal is not None else ""
    )
    ax_gantt.set_title(
        f"Schedule – Gantt Chart    ({makespan_label})",
        fontweight="bold",
        fontsize=13,
    )

    # ── Resource utilization per time slot ────────────────────────────────────
    for k in range(K):
        ax_res = axes[1 + k]
        usage = np.zeros(T_norm, dtype=float)

        for i, (m, t_start, t_end) in schedule.items():
            for t in range(t_start, t_end):
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
        ax_res.set_title(f"Resource {k} utilization", fontweight="bold", fontsize=11)
        ax_res.set_xlabel("Time (normalized)", fontsize=10)
        ax_res.set_ylabel("Usage", fontsize=10)
        ax_res.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_res.legend(fontsize=9)
        ax_res.grid(axis="y", linestyle="--", alpha=0.4)
        ax_res.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    plt.savefig(f"plots/{filename}.png")
