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
from sbi.inference import NPE
from sbi.utils import BoxUniform


def rep_parameters(parameters, parameters_to_calibrate, draw):
    sim_paramerters = parameters.copy()
    for param, value in zip(parameters_to_calibrate, draw.tolist()):
        sim_paramerters[param] = value
    return sim_paramerters


def run_sim(parameters, initial_conditions, T, keys):
    data = jl.run_simulation(parameters, initial_conditions, T, keys)
    return np.array(data)


def run_monte_carlo(parameters, initial_conditions, T, num_simulations, keys):
    data = jl.run_monte_carlo(parameters, initial_conditions, T, num_simulations, keys)
    data = np.mean(data, axis=0)  # average across simulations
    return np.array(data)


### initial calibration
initial_calibration = datetime(2010, 3, 31)

n = 3  # number of years to simulate -> 3 years
T_hist = 4 * n - 1  # quaters
final_calibration = datetime(
    initial_calibration.year + T_hist // 4, 12, initial_calibration.day
)

T_forecast = 4 * 3  # 3 years forecast
date_forecast = datetime(
    final_calibration.year + T_forecast // 4,
    final_calibration.month,
    final_calibration.day,
)
### real data

quarterly_dates, bit = [np.array(x) for x in jl.get_real(["anc"])]

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

# parameters_to_calibrate = ["psi", "alpha_G", "beta_E", "alpha_E", "xi_gamma"]
parameters_to_calibrate = ["theta", "zeta", "zeta_LTV", "zeta_b"]

priors_bounds = [
    (0.01, 0.99),  # theta
    (0.01, 0.5),  # zeta
    (0.1, 1.2),  # zeta_LTV
    (0, 2),  # zeta_B
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

avaible_keys = [
    "real_gdp",
    "gdp_deflator",

]

def run_prior_check():
    pass

def gen_sample (calibration_date, T, priors, params_to_calibrate, n_samples, n_runs, keys):
    theta_draws = priors.sample((n_samples,))
    params, initial_conditions = jl.calibrate(
        calibration_date.year, calibration_date.month, calibration_date.day
    )
    samples = np.zeros((n_samples, n_runs, len(keys), T + 1))
    for i, draw in enumerate(theta_draws):
        sim_parameters = rep_parameters(params, params_to_calibrate, draw)
        sim_out = jl.run_monte_carlo(sim_parameters, initial_conditions, T, n_runs, keys)
        sim_out = np.array(sim_out)
        samples[i, :, :, :] = sim_out
    if "prior_check" in argv:
        run_prior_check()
    return theta_draws.numpy(), samples

num_calibrations = 4
n_runs = 10
keys = ["real_gdp", "gdp_deflator"]
samples = np.zeros((n_histories, num_calibrations, n_runs, len(keys), T_hist + 1))
prior_draws = np.zeros((num_calibrations, n_histories, len(parameters_to_calibrate)))

for i in range(4):
    cal_date_year = initial_calibration.year + i
    cal_date = datetime(cal_date_year, initial_calibration.month, initial_calibration.day)
    theta_draws, sample = gen_sample(
        calibration_date=cal_date,
        T=T_hist,
        priors=priors,
        params_to_calibrate=parameters_to_calibrate,
        n_samples=n_histories,
        n_runs=n_runs,
        keys=keys,
    )
    samples[:, i, :, :] = sample
    prior_draws[i, :, :] = theta_draws
    print(f"Calibration date: {cal_date}, sample shape: {sample.shape}")

### save
np.savez(
    f"data/prior_samples_n{n_histories}_{','.join(parameters_to_calibrate)}_bounds{','.join([str(b) for b in priors_bounds.numpy()])}.npz",
    sim_out=samples,
    theta_draw = prior_draws,
    bounds = priors_bounds.numpy(),
    parameters_to_calibrate = parameters_to_calibrate
)

