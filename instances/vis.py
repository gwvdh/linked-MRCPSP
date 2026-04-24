from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# PhaseTimeline[time][resource_index] = count
PhaseTimeline = Dict[int, Dict[int, int]]

_CMAP_NAME = "tab10"  # supports up to 10 distinct colours


def _build_res_conf(resource_indices: List[int]) -> Dict[int, Dict[str, str]]:
    """
    Build a display-config dict for an arbitrary set of resource indices.

    :param resource_indices: Sorted list of integer resource indices found in the data.
    :returns: ``{resource_index: {"color": ..., "label": ...}}``
    """
    cmap = plt.get_cmap(_CMAP_NAME)
    n = max(len(resource_indices), 1)
    return {
        idx: {
            "color": cmap(i / n),
            "label": f"Level {idx + 1}",
        }
        for i, idx in enumerate(resource_indices)
    }


def plot_timelines(
    phase_timelines: List[PhaseTimeline],
    filename: str = "timelines.png",
    output_dir: str = "plots",
) -> None:
    """
    Visualise resource usage per phase using stepped line charts.

    :param phase_timelines: ``phase_timelines[phase][time][resource_index]``
    :param filename: Output filename (saved inside *output_dir*)
    :param output_dir: Directory to write the plot into (created if absent)
    """
    all_times = [t for tl in phase_timelines for t in tl]
    if not all_times:
        print("No simulation data to plot.")
        return

    # Collect every resource index that appears across all phases
    all_resource_indices: List[int] = sorted(
        {
            res
            for tl in phase_timelines
            for res_counts in tl.values()
            for res in res_counts
        }
    )
    res_conf = _build_res_conf(all_resource_indices)

    min_t, max_t = min(all_times), max(all_times)
    time_range = list(range(min_t, max_t + 2))

    num_phases = len(phase_timelines)
    fig, axes = plt.subplots(
        num_phases,
        1,
        figsize=(12, 4 * num_phases),
        sharex=True,
    )
    if num_phases == 1:
        axes = [axes]

    for i, (ax, timeline) in enumerate(zip(axes, phase_timelines)):
        max_y = 0

        for res_idx, conf in res_conf.items():
            y_values = [
                timeline.get(t, {}).get(res_idx, 0) for t in time_range[:-1]
            ] + [0]
            max_y = max(max_y, max(y_values))

            ax.step(
                time_range,
                y_values,
                where="post",
                color=conf["color"],
                label=conf["label"],
                linewidth=2,
            )
            ax.fill_between(
                time_range,
                y_values,
                step="post",
                color=conf["color"],
                alpha=0.1,
            )

        ax.set_title(f"Phase {i + 1} Resource Demand")
        ax.set_ylabel("Units Required")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(0, max_y + 1.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    plt.xlabel("Time (Simulation Ticks)")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
