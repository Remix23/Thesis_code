import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

from os import path, listdir

post_dir = "posterior_stats"

files = listdir(post_dir)

def compute_statistic (data, p):
    ### data: (n_samples, n_parameters)
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    q1 = np.quantile(data, p, axis=0)
    q3 = np.quantile(data, 1 - p, axis=0)
    return median, std, q1, q3

### npe
npe_embeddings = {}

nre_embeddings = {}

seeds = set()

### find all seeds and embeddings
for f in files:
    split = f[:-4].split("_")
    if "seed" not in split:
        print(f"Warning: file {f} does not contain a seed identifier.")
        continue

    seed_idx = split.index("seed")
    seed = int(split[seed_idx + 1])
    seeds.add(seed)

    if "npe" in f:
        embedding_idx = split.index("npe")
        embedding = split[embedding_idx + 1]
        if embedding not in npe_embeddings:
            npe_embeddings[embedding] = []

        npe_embeddings[embedding].append(f)

    if "nre" in f:
        embedding_idx = split.index("nre")
        embedding = split[embedding_idx + 1]
        if embedding not in nre_embeddings:
            nre_embeddings[embedding] = []

        nre_embeddings[embedding].append(f)

### check whether we have seeds for all embeddings
for embedding, files in npe_embeddings.items():
    if len(files) != len(seeds):
        print(f"Warning: NPE embedding {embedding} has {len(files)} files, but there are {len(seeds)} seeds.")

for embedding, files in nre_embeddings.items():
    if len(files) != len(seeds):
        print(f"Warning: NRE embedding {embedding} has {len(files)} files, but there are {len(seeds)} seeds.")

print(f"Found {len(seeds)} seeds: {seeds}")

### computing seed - ensemble statistics for NPE and NRE embeddings
for embedding, files in npe_embeddings.items():
    medians = np.zeros((len(files), 3))  # (n_seeds, n_parameters)
    int_widths = np.zeros((len(files), 3))  # (n_seeds, n_parameters)

    for i, f in enumerate(files):
        df = pd.read_csv(path.join(post_dir, f))
        data = df.to_numpy()
        stats = compute_statistic(data, 0.05) # (median, std, q1, q3)
        medians[i, :] = stats[0]
        int_widths[i, :] = stats[3] - stats[2]

    ensemble_median = np.median(medians, axis=0)
    mins_medians, maxs_medians = np.min(medians, axis=0), np.max(medians, axis=0)
    print(f"For NPE embedding {embedding}:")
    print(f"  Ensemble median: {" & ".join([f'{m:.2f} ({l:.2f}-{u:.2f})' for m, l, u in zip(ensemble_median, mins_medians, maxs_medians)])}")
    print(f"  Average interval width: {" & ".join([f'{w:.2f}' for w in np.mean(int_widths, axis=0)])}")
