from datetime import date, datetime
from os import environ, getcwd, listdir, path
from sys import argv

NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl

jl.seval("using JuliaModel: run_simulation, get_real, calibrate, run_monte_carlo")
print(f"Using JuliaModel with {jl.seval('Threads.nthreads()')} threads.")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.analysis import pairplot
from sbi.diagnostics import check_sbc, run_sbc
from sbi.inference import NPE
from sbi.utils import BoxUniform


def rep_parameters(parameters, parameters_to_calibrate, draw):
    sim_paramerters = parameters.copy()
    for param, value in zip(parameters_to_calibrate, draw.tolist()):
        sim_paramerters[param] = value
    return sim_paramerters


def run_sim(parameters, initial_conditions, T):
    data = jl.run_simulation(parameters, initial_conditions, T)
    return np.array(data)


def run_monte_carlo(parameters, initial_conditions, T, num_simulations):
    data = jl.run_monte_carlo(parameters, initial_conditions, T, num_simulations)
    data = np.mean(data, axis=0)  # average across simulations
    return np.array(data)


### initial calibration
initial_calibration = datetime(2010, 3, 31)

n = 6  # number of years to simulate -> 5 years
T_hist = 4 * n - 1  # quaters
final_calibration = datetime(
    initial_calibration.year + T_hist // 4, 12, initial_calibration.day
)

T_forecast = 4 * 4  # 5 years forecast
date_forecast = datetime(
    final_calibration.year + T_forecast // 4,
    final_calibration.month,
    final_calibration.day,
)
### real data
csv_data = pd.read_csv("data_npz/italy_real_gdp.csv", parse_dates=["observation_date"])
csv_data = csv_data[csv_data["observation_date"] >= initial_calibration]
quarterly_dates, bit = [np.array(x) for x in jl.get_real()]

n = len(quarterly_dates)
df = pd.DataFrame({"date": quarterly_dates.flatten(), "real": bit.flatten()})
df = df[df["date"] >= initial_calibration]
df = df[df["date"] <= date_forecast]

hist_params, hist_initial = jl.calibrate(
    initial_calibration.year, initial_calibration.month, initial_calibration.day
)

final_params, final_initial = jl.calibrate(
    final_calibration.year, final_calibration.month, final_calibration.day
)
### gen historical date for NPE
beta0 = hist_params["beta_E"]

# parameters_to_calibrate = ["psi", "alpha_G", "beta_E", "alpha_E", "xi_gamma"]
parameters_to_calibrate = ["psi", "alpha_G", "beta_E", "alpha_E"]

priors_bounds = [
    (0.9, 0.99),  # psi
    (0.9, 0.99),  # alpha_G,
    (0.7*beta0, 1.3 * beta0), # beta_E
    (0.9, 0.99),  # alpha_E
    # (0.8, 0.99),  # rho
]

for param, bounds in zip(parameters_to_calibrate, priors_bounds):
    calibrated = hist_params[param]
    if calibrated <= bounds[0] or calibrated >= bounds[1]:
        print(
            f"Warning: Parameter {param} with value {calibrated} is outside the bounds {bounds}"
        )
    else:
        print(
            f"Parameter {param} with value {calibrated} is within the bounds {bounds}"
        )

    print(f"new: {final_params[param]}, old: {hist_params[param]}")

n_histories = int(
    input(
        "Enter the number of historical trajectories to generate for NPE data generation: "
    )
)
priors_bounds = torch.tensor(priors_bounds, dtype=torch.float32)
priors = BoxUniform(low=priors_bounds[:, 0], high=priors_bounds[:, 1])

raw = np.ndarray(shape=(n_histories, T_hist + 1))

if "prior_check" in argv:
    plt.figure(figsize=(12, 8))

priors_samples = priors.sample((n_histories,))
for i, draw in enumerate(priors_samples):
    sim_parameters = rep_parameters(hist_params, parameters_to_calibrate, draw)
    sim_initial = hist_initial
    sim_out = run_monte_carlo(sim_parameters, sim_initial, T_hist, num_simulations=15)
    if "prior_check" in argv:
        plt.plot(sim_out, alpha=0.2, color="blue")
    raw[i, :] = sim_out
    print(
        f"Completed {i + 1}/{n_histories} simulations for NPE data generation", end="\r"
    )
print()

### saving
np.savez(
    "npe_data.npz",
    priors_samples=priors_samples,
    raw=raw,
    parameters_to_calibrate=parameters_to_calibrate,
    bounds=priors_bounds,
)

if "prior_check" in argv:
    plt.plot(df["real"].values[: T_hist + 1], color="black", label="Real GDP")
    plt.title("Simulated trajectories from the prior")
    plt.xlabel("Time (quarters)")
    plt.ylabel("Real GDP")
    plt.savefig(
        f"pngs/prior_check_n{n_histories}_{', '.join(parameters_to_calibrate)}_bounds{', '.join([str(b) for b in priors_bounds.numpy()])}.png"
    )
    plt.show()
