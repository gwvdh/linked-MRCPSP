from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PhaseTimeline = Dict[int, Dict[int, int]]
PHASE_CMAP = "tab10"


def plot_combined_resource_demands(
    timelines_by_resource: list[list[PhaseTimeline]],
    resource_indices: list[int],
    filename: str = "resource_demands.png",
    output_dir: str = "plots",
) -> None:
    """
    Plot, for each resource, total demand over time and its decomposition by phase.
    """
    if not timelines_by_resource:
        print("No timeline data to plot.")
        return

    n_resources = len(resource_indices)
    n_phases = len(timelines_by_resource[0])
    cmap = plt.get_cmap(PHASE_CMAP)
    phase_colors = [cmap(i / max(n_phases, 1)) for i in range(n_phases)]

    fig, axes = plt.subplots(
        n_resources,
        1,
        figsize=(14, 4 * n_resources),
        squeeze=False,
    )

    for row, (resource_idx, resource_timelines) in enumerate(
        zip(resource_indices, timelines_by_resource)
    ):
        ax = axes[row, 0]

        all_times = sorted({t for tl in resource_timelines for t in tl})
        if not all_times:
            ax.set_visible(False)
            continue

        time_points = list(range(min(all_times), max(all_times) + 2))

        phase_profiles: list[list[int]] = [
            [phase_tl.get(t, {}).get(resource_idx, 0) for t in time_points[:-1]] + [0]
            for phase_tl in resource_timelines
        ]

        totals = [
            sum(profile[t] for profile in phase_profiles)
            for t in range(len(time_points))
        ]

        bottoms = [0] * len(time_points)
        for phase_idx, profile in enumerate(phase_profiles):
            tops = [b + x for b, x in zip(bottoms, profile)]

            ax.step(
                time_points,
                tops,
                where="post",
                color=phase_colors[phase_idx],
                linewidth=1.0,
                alpha=0.7,
            )
            ax.fill_between(
                time_points,
                bottoms,
                tops,
                step="post",
                color=phase_colors[phase_idx],
                alpha=0.35,
                label=f"Phase {phase_idx + 1}",
            )
            bottoms = tops

        ax.step(
            time_points,
            totals,
            where="post",
            color="black",
            linewidth=2.0,
            label="Total",
        )

        ax.set_title(f"Resource {resource_idx + 1} — Combined Demand")
        ax.set_xlabel("Time")
        ax.set_ylabel("Required units")
        ax.set_ylim(0, max(totals, default=0) + 1.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.95)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
