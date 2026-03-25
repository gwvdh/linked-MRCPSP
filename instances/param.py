from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=8)
samples = sampler.random(n=200)

param_bounds = {
    "arrival_rate": (0.2, 1.0),
    "batch_size": (1, 10),
    "max_base_duration": (1.0, 10.0),
    "resource_1_ratio_center": (1.0, 3.0),
    "resource_1_ratio_spread": (0.0, 1.0),
    "resource_2_ratio_center": (1.0, 3.0),
    "resource_2_ratio_spread": (0.0, 1.0),
    "res_1_2_multiplier": (1.0, 3.0),
    "res_1_3_multiplier": (1.5, 4.0)
}


