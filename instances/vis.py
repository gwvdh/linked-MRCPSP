from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# PhaseTimeline[time][resource_index] = count
PhaseTimeline = Dict[int, Dict[int, int]]

_PHASE_CMAP = "tab10"


def plot_combined_resource_demands(
    timelines_by_resource: List[List[PhaseTimeline]],
    resource_indices: List[int],
    filename: str = "resource_demands.png",
    output_dir: str = "plots",
) -> None:
    """
    One axes per resource, all combined into a single figure. Each axes shows
    demand across all phases over time: phase contributions as stacked filled
    step areas, total demand as a bold line.

    :param timelines_by_resource: ``[resource][phase][time][res_idx] = count``
        Outer list aligned with *resource_indices*; inner list is one
        PhaseTimeline per phase.
    :param resource_indices: Integer resource indices for each entry of
        *timelines_by_resource* (used for axis labels).
    :param filename: Output filename saved inside *output_dir*.
    :param output_dir: Directory to write the plot into (created if absent).
    """
    if not timelines_by_resource:
        print("No timeline data to plot.")
        return

    n_resources = len(resource_indices)
    n_phases = len(timelines_by_resource[0])
    cmap = plt.get_cmap(_PHASE_CMAP)
    phase_colors = [cmap(i / max(n_phases, 1)) for i in range(n_phases)]

    fig, axes = plt.subplots(
        n_resources,
        1,
        figsize=(14, 4 * n_resources),
        sharex=False,
        squeeze=False,
    )

    for row, (res_idx, res_timelines) in enumerate(
        zip(resource_indices, timelines_by_resource)
    ):
        ax = axes[row, 0]

        all_times = sorted({t for tl in res_timelines for t in tl})
        if not all_times:
            ax.set_visible(False)
            continue

        time_range = list(range(min(all_times), max(all_times) + 2))

        phase_demands: List[List[int]] = [
            [phase_tl.get(t, {}).get(res_idx, 0) for t in time_range[:-1]] + [0]
            for phase_tl in res_timelines
        ]

        totals = [
            sum(pd[i] for pd in phase_demands) for i in range(len(time_range))
        ]

        bottoms = [0] * len(time_range)
        for phase_id, pd in enumerate(phase_demands):
            tops = [b + d for b, d in zip(bottoms, pd)]
            ax.step(
                time_range,
                tops,
                where="post",
                color=phase_colors[phase_id],
                linewidth=1.0,
                alpha=0.6,
            )
            ax.fill_between(
                time_range,
                bottoms,
                tops,
                step="post",
                color=phase_colors[phase_id],
                alpha=0.35,
                label=f"Phase {phase_id + 1}",
            )
            bottoms = tops

        ax.step(
            time_range,
            totals,
            where="post",
            color="black",
            linewidth=2.0,
            label="Total",
        )

        ax.set_title(f"Resource {res_idx + 1} — Combined Demand Across All Phases")
        ax.set_xlabel("Time")
        ax.set_ylabel("Units Required")
        ax.set_ylim(0, max(totals, default=0) + 1.5)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
