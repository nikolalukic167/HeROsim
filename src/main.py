import numpy as np
from scipy.stats import qmc

weights = [0.6, 0.3, 0.1]  # Priority for params A, B, C
sampler = qmc.LatinHypercube(d=3, optimization="random-cd")
samples = sampler.random(n=1000)
weighted_samples = samples * np.array(weights)  # Scale by parameter weights
pass
