import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

# --- Feature extraction ---

def order_strength(precedences, n_jobs):
    max_arcs = n_jobs * (n_jobs - 1) / 2
    return len(precedences) / max_arcs if max_arcs else 0.0

def proc_time_variance(modes):
    return float(np.var([m["duration"] for job in modes for m in job]))

def demand_ratio(modes, n_resources):
    D = np.array([m["demands"] for job in modes for m in job])
    avg = D.mean(axis=0)
    return float(avg.min() / avg.max()) if avg.max() else 0.0

def compute_features(inst):
    return np.array([
        order_strength(inst["precedences"], inst["n_jobs"]),
        inst["resource_strength"],
        proc_time_variance(inst["modes"]),
        demand_ratio(inst["modes"], inst["n_resources"]),
    ])

def feature_matrix(instances):
    return np.vstack([compute_features(i) for i in instances])

# --- Plot ---

def plot_instance_space(real_F, gen_F, caption=""):
    """Plots the feature space of real and generated instances."""
    all_F = np.vstack([real_F, gen_F])
    scaled = StandardScaler().fit_transform(all_F)

    pca = PCA(n_components=2).fit(scaled)
    r2d, g2d = pca.transform(scaled[:len(real_F)]), pca.transform(scaled[len(real_F):])
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for pts, c in [(g2d, "orange"), (r2d, "steelblue")]:
        if len(pts) >= 3:
            h = ConvexHull(pts)
            v = np.append(h.vertices, h.vertices[0])
            ax.fill(pts[h.vertices, 0], pts[h.vertices, 1], color=c, alpha=0.15)
            ax.plot(pts[v, 0], pts[v, 1], color=c, lw=1.2)

    ax.scatter(*g2d.T, marker="x", color="orange",    s=40, lw=1.2, label="Generated", zorder=3)
    ax.scatter(*r2d.T, marker="o", color="steelblue", s=30,         label="Real",      zorder=4)
    ax.set_xlabel(f"PC1 ({var[0]:.2f}% explained variance)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]:.2f}% explained variance)", fontsize=9)
    ax.legend(fontsize=8)
    if caption:
        fig.text(0.5, -0.04, caption, ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    return fig, ax

# --- Synthetic data & demo ---

def make_instance(rng, n_jobs=None, rs=None, n_modes=3, n_resources=2, dense=False):
    n_jobs = n_jobs or int(rng.integers(5, 20))
    caps = rng.integers(5, 20, size=n_resources).tolist()
    return {
        "n_jobs": n_jobs,
        "n_resources": n_resources,
        "resource_strength": rs if rs is not None else float(rng.uniform(0, 1)),
        "precedences": [(i, j) for i in range(n_jobs) for j in range(i+1, n_jobs)
                        if rng.random() < (0.4 if dense else 0.1)],
        "modes": [[{"duration": float(rng.integers(1, 20)),
                    "demands": [float(rng.integers(1, c)) for c in caps]}
                   for _ in range(n_modes)] for _ in range(n_jobs)],
    }

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    real = [make_instance(rng, n_jobs=int(rng.integers(10,15)), rs=float(rng.uniform(0.2,0.5)), dense=True)  for _ in range(20)]
    gen  = [make_instance(rng, n_modes=int(rng.integers(2, 5)))                                               for _ in range(150)]

    fig, _ = plot_instance_space(
        feature_matrix(real), feature_matrix(gen)
    )
    plt.savefig("instance_space.png", dpi=150, bbox_inches="tight")
    plt.show()
