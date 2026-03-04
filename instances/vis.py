import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .definitions import ResourceLevel

def plot_timelines(phase_timelines: List[Dict[int, Dict[ResourceLevel, int]]], scarcity=0.5, filename: str = "timelines.png"):
    """
    Visualizes resource usage per phase using Stepped Line Charts.
    """
    # 1. Determine global time range
    all_times = [t for tl in phase_timelines for t in tl.keys()]
    if not all_times:
        print("No simulation data to plot.")
        return
    
    min_t, max_t = min(all_times), max(all_times)
    # Add a buffer to the end for the 'post' step visual
    time_range = list(range(min_t, max_t + 2)) 
    
    # 2. Setup Subplots (One per Phase)
    num_phases = len(phase_timelines)
    fig, axes = plt.subplots(num_phases, 1, figsize=(12, 4 * num_phases), sharex=True)
    if num_phases == 1: axes = [axes]

    # Distinct colors
    res_conf = {
        ResourceLevel.L1: {'color': '#1f77b4', 'label': 'Level 1'}, # Blue
        ResourceLevel.L2: {'color': '#ff7f0e', 'label': 'Level 2'}, # Orange
        ResourceLevel.L3: {'color': '#2ca02c', 'label': 'Level 3'}  # Green
    }

    for i, (ax, timeline) in enumerate(zip(axes, phase_timelines)):
        max_y = 0
        
        for res in [ResourceLevel.L1, ResourceLevel.L2, ResourceLevel.L3]:
            # Extract data, filling gaps with 0
            # Note: We append 0 at the end to close the step chart visually
            y_values = [timeline.get(t, {}).get(res, 0) for t in time_range[:-1]] + [0]
            max_y = max(max_y, max(y_values))
            
            # Plot Stepped Line
            ax.step(time_range, y_values, where='post', 
                    color=res_conf[res]['color'], 
                    label=res_conf[res]['label'], 
                    linewidth=2)
            
            # Light fill to show volume without obscuring other lines
            ax.fill_between(time_range, y_values, step='post', 
                            color=res_conf[res]['color'], 
                            alpha=0.1)

        ax.set_title(f"Phase {i+1} Resource Demand")
        ax.set_ylabel("Units Required")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, max_y + 1.5) # Add top margin
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Force integer ticks

        if i == 0: ax.legend(loc='upper right', framealpha=0.9)

    plt.xlabel("Time (Simulation Ticks)")
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
