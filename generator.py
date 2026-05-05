from os import environ, listdir, path, getcwd
from sys import argv

NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl

jl.seval("using JuliaModel: run_simulation, get_parameters, get_initial_conditions, run_for_different_parameters, run_monte_carlo")
print(f"Using JuliaModel with {jl.seval('Threads.nthreads()')} threads.")

from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.analysis import pairplot

from sbi.diagnostics import check_sbc, run_sbc

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import time

CALIBRATION_DATE = "2010-01-01"
T = 20

STATISTISC = {
    "mean_growth": True,
    "std_growth": True,
    "yearly_corr": True,
    "ar_1": True,
    "ar_2": True,
    "skewness_gdp": True
}

def rep_parameters(parameters, params_to_calibrate, draw):
    sim_paramerters = parameters.copy()
    for param, value in zip(params_to_calibrate, draw.tolist()):
        sim_paramerters[param] = value
    return sim_paramerters

def run_sim(parameters, initial_conditions, T):
    data = jl.run_simulation(parameters, initial_conditions, T)
    return np.array(data)

def run_monte_carlo(parameters, initial_conditions, T, num_simulations):
    data = jl.run_monte_carlo(parameters, initial_conditions, T, num_simulations)
    data = np.mean(data, axis=0) # average across simulations
    return np.array(data)

def ar_params(time_series, p):
    """Estimate AR(p) parameters for a univariate time series using OLS.

    Returns array of length p with AR coefficients (phi_1 ... phi_p) and
    intercept as a separate value: (intercept, coeffs_array).
    """
    y = np.asarray(time_series)
    n = y.shape[0]
    if n <= p:
        raise ValueError("Time series length must be greater than p")

    # build lagged design matrix with intercept
    X = np.ones((n - p, p + 1))
    for i in range(p):
        X[:, i + 1] = y[p - i - 1:n - i - 1]
    y_target = y[p:]

    # OLS solution
    coef, *_ = np.linalg.lstsq(X, y_target, rcond=None)
    intercept = coef[0]
    ar_coefs = coef[1:]
    return ar_coefs[p - 1], ar_coefs

def compute_statistics (sim_out, chosen_stats):
    # compute summary statistics from the simulated data
    # for example, we can compute the mean and standard deviation of real GDP
    log_diff = np.diff(np.log(sim_out))

    mean_gdprowth = np.mean(log_diff)

    std_gdprowth = np.std(log_diff)

    ### yearly correlation y_t and y_t-4
    yearly_corr = np.corrcoef(log_diff[:-4], log_diff[4:])[0, 1]
    ### AR(1) coefficient of real GDP
    ar1_coeff = ar_params(log_diff, 1)[0]
    ar2_coeff = ar_params(log_diff, 2)[0]
    ### skewness of real GDP
    skewness_gdp = np.mean((log_diff - mean_gdprowth)**3) / std_gdprowth**3

    res = {
        "mean_growth": mean_gdprowth,
        "std_growth": std_gdprowth,
        "yearly_corr": yearly_corr,
        "ar_1": ar1_coeff,
        "ar_2": ar2_coeff,
        "skewness_gdp": skewness_gdp
    }
    return np.array([res[stat] for stat in chosen_stats])


parameters = jl.get_parameters()

initial_conditions = jl.get_initial_conditions()

PARAMETERS = [
    "psi", # consumption parameter
    "alpha_G", # capital share in production function
    "alpha_E",
    "beta_E", # depreciation rate of capital
    "xi_gamma"
]

beta0 = parameters["beta_E"]

priors_bounds = {
    "psi": (0.7, 0.99),
    "alpha_G": (0.7, 0.99),
    "alpha_E": (0.1, 0.5),
    "beta_E": (0.5*beta0, 1.5 * beta0),
    "xi_gamma": (0.7, 1.5)
}
print("Parameters to calibrate:")
for i, param in enumerate(PARAMETERS):
    print(f"{i}: {param}")

choosen = input("Enter the numbers of the parameters to calibrate, separated by commas (e.g., 0,2,4): ")
choosen_indices = [int(x.strip()) for x in choosen.split(",")]

parameters_to_calibrate = [PARAMETERS[i] for i in choosen_indices]
print(f"Chosen parameters to calibrate: {parameters_to_calibrate}")

bounds = torch.tensor([priors_bounds[param] for param in parameters_to_calibrate], dtype=torch.float32)
prior = BoxUniform(low=bounds[:, 0], high=bounds[:, 1])

n_draws = int(input("Enter the number of prior draws to generate (e.g., 1000): "))
npe_x = np.ndarray(shape = (n_draws, T + 1))
priors_draws = prior.sample((n_draws,))

if not "raw" in argv:
### statistics
    avaiable_stats = ["mean_growth", "std_growth", "ar_1", "ar_2", "yearly_corr", "skewness_gdp"]

    print("Available summary statistics:")
    for i, stat in enumerate(avaiable_stats):
        print(f"{i}: {stat}")
    choosen_stats = input("Enter the numbers of the summary statistics to use, separated by commas (e.g., 0,2,4): ")
    choosen_stats_indices = [int(x.strip()) for x in choosen_stats.split(",")]
    chosen_stats = [avaiable_stats[i] for i in choosen_stats_indices]
    print(f"Chosen summary statistics: {chosen_stats}")
    npe_stats = np.ndarray(shape = (n_draws, len(chosen_stats)))

for i, draw in enumerate(priors_draws):
    sim_paramerters = rep_parameters(parameters, parameters_to_calibrate, draw)
    x = run_monte_carlo(sim_paramerters, initial_conditions, T, num_simulations=10)
    npe_x[i, :] = x
    if not "raw" in argv:
        npe_stats[i, :] = compute_statistics(x, chosen_stats)
    print(f"Generated simulation {i+1}/{n_draws}", end="\r")

filename = f"sim_data_{n_draws}_draws.npz"

if "raw" in argv:
    np.savez(filename, prior_draws=priors_draws.numpy(), raw=npe_x, to_calibrate = parameters_to_calibrate, bounds = bounds.numpy())
    exit()
    
filename = f"sim_data_{n_draws}_draws_stats.npz"
np.savez(filename, prior_draws=priors_draws.numpy(), raw=npe_x, to_calibrate = parameters_to_calibrate, bounds = bounds.numpy(), chosen_stats = chosen_stats)
