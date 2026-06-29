import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from os import path, listdir

T = [1, 2, 3, 4, 8, 12]

### the single forecast on log - diff
dir_path = "rmsfe3"

files = listdir(dir_path)

keys = [
        "real_gdp_quarterly",
        "gdp_deflator_quarterly",
        "real_household_consumption_quarterly",
        "real_government_consumption_quarterly",
        "real_capitalformation_quarterly",
    ]

ref = list(filter(lambda f: "ar1" in f.lower(), files))[0]
idx_ar = files.index(ref)

abm = list(filter(lambda f: "abm" in f.lower(), files))[0]
idx_abm = files.index(abm)

rsmfes = np.zeros((len(files), len(T), 5))  # (n_models, n_forecast_horizons, n_features)

for i, f in enumerate(files):
    if f.endswith(".csv"):
        data = np.loadtxt(path.join(dir_path, f), delimiter=",", skiprows=1)
        rsmfes[i, :, :] = data

abm_rmsfe = rsmfes[idx_abm, :, :]
ar_rmsfe = rsmfes[idx_ar, :, :]

improvement_ar = np.zeros((len(files), len(T), len(keys)))
improvemnt_abm = np.zeros((len(files), len(T), len(keys)))
for i, rsmfe in enumerate(rsmfes): ### across models
    # if i in [idx_ar]: continue

    ### comparison to ABM
    per_improvement_abm = (abm_rmsfe - rsmfe) / abm_rmsfe
    improvemnt_abm[i, :, :] = per_improvement_abm
    ### comparison to AR(1) 
    per_improvement_ar = (ar_rmsfe - rsmfe) / ar_rmsfe
    improvement_ar[i, :, :] = per_improvement_ar
    print (f"Model {files[i]}:")
    for feature in range(rsmfes.shape[2]):
        print(f"Feature {keys[feature]}:")
        for j, t in enumerate(T):
            print(f"  T={t}: Improvement over ABM: {per_improvement_abm[j, feature]:.4f}, Improvement over AR(1): {per_improvement_ar[j, feature]:.4f}")

### printing the table
to_names = {
    "real_gdp_quarterly" : "GDP",
    "gdp_deflator_quarterly" : "Inflation",
    "real_household_consumption_quarterly" : "Household Cons.",
    "real_government_consumption_quarterly" : "Government Cons.",
    "real_capitalformation_quarterly" : "Investment",
}

methods_NPE = ["m,s", "seq_multivariate", "simple", "hierarchical"]
for method in methods_NPE:
    idx = list(filter(lambda f: method in f.lower(), files))[0]
    idx = files.index(idx)
    print(f"Method {method}:")
    for j, key in enumerate(keys):
        print(f"& {to_names[key]} ", end="")
        for t in range(len(T)):
            print(f" & {improvement_ar[idx, t, j]*100:.2f}\\% ", end="")
        print("\\\\")

for j, key in enumerate(keys):
    print(f"& {to_names[key]} ", end="")
    for t in range(len(T)):
        print(f" & {ar_rmsfe[t, j] * 10**3:.2f} $\\cdot 10^{{-3}}$ ", end="")
        # if key == "gdp_deflator_quarterly":
        #     print(f" & {abm_rmsfe[t, j] * 10**(3):.2f} $\\cdot 10^{{-3}}$ ", end="")
        #     continue
        # print(f" & {abm_rmsfe[t, j] * 10**(-2):.2f} $\\cdot 10^{{2}}$", end="")
    print("\\\\")