from os import path, listdir, getcwd, environ
from sys import argv
from datetime import datetime, date
NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl
jl.seval("using JuliaModel: run_simulation, get_real, calibrate, run_monte_carlo")
print(f"Using JuliaModel with {jl.seval('Threads.nthreads()')} threads.")

from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.analysis import pairplot

from sbi.diagnostics import check_sbc, run_sbc

import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


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
    data = np.mean(data, axis=0) # average across simulations
    return np.array(data)

def findar_p (times_series, p):
    target = times_series[p:]
    lagged = np.array([times_series[i:-(p - i)] for i in range(p)]).T
    ones = np.ones((lagged.shape[0], 1))
    lagged = np.hstack((ones, lagged))
    ols = np.linalg.lstsq(lagged, target, rcond=None)
    return ols[0]

def compute_statistics (sim_out):
    # compute summary statistics from the simulated data
    # for example, we can compute the mean and standard deviation of real GDP
    growth_rates = np.diff(np.log(sim_out))

    mean_gdp = np.mean(sim_out)
    mean_gdprowth = np.mean(growth_rates)

    std_gdp = np.std(sim_out)
    std_gdprowth = np.std(growth_rates)

    ### yearly correlation y_t and y_t-4
    yearly_corr = np.corrcoef(growth_rates[:-4], growth_rates[4:])[0, 1]
    ### AR(1) coefficient of real GDP
    ar1_coeff = findar_p(sim_out, 1)[1]
    ar2_coeff = findar_p(sim_out, 2)[2]
    ### skewness of real GDP
    skewness_gdp = np.mean((growth_rates - mean_gdprowth)**3) / std_gdprowth**3
    return np.array([mean_gdprowth, std_gdprowth, yearly_corr, ar1_coeff, ar2_coeff, skewness_gdp])


### initial calibration
initial_calibration = datetime(2010, 3, 31)

n = 6 # number of years to simulate -> 5 years
T_hist = 4 * n - 1 # quaters
final_calibration = datetime(initial_calibration.year + T_hist // 4, 12, initial_calibration.day)

T_forecast = 4 * 4 # 5 years forecast
date_forecast = datetime(final_calibration.year + T_forecast // 4, final_calibration.month, final_calibration.day)
### real data
csv_data = pd.read_csv("data/italy_real_gdp.csv", parse_dates=["observation_date"])
csv_data = csv_data[csv_data["observation_date"] >= initial_calibration]
quarterly_dates, bit = [np.array(x) for x in jl.get_real()]

n = len(quarterly_dates)
df = pd.DataFrame({"date": quarterly_dates.flatten(), "real": bit.flatten()})
df = df[df["date"] >= initial_calibration]
df = df[df["date"] <= date_forecast]

hist_params, hist_initial = jl.calibrate(initial_calibration.year, initial_calibration.month, initial_calibration.day)

final_params, final_initial = jl.calibrate(final_calibration.year, final_calibration.month, final_calibration.day)
### gen historical date for NPE
beta0 = hist_params["beta_E"]

# parameters_to_calibrate = ["rho", "alpha_G", "alpha_E", "beta_E", "xi_gamma"]
parameters_to_calibrate = ["psi", "alpha_G", "beta_E", "xi_gamma"]

priors_bounds = [
    (0.7, 0.99), # psi
    (0.8, .999), # alpha_G,
    (0.5*beta0, 1.5 * beta0), # beta_E
    # (0.5, .99), # alpha_E
    (.7, 1.5),    # xi_gamma
]

for param, bounds in zip(parameters_to_calibrate, priors_bounds):
    calibrated = hist_params[param]
    if calibrated <= bounds[0] or calibrated >= bounds[1]:
        print(f"Warning: Parameter {param} with value {calibrated} is outside the bounds {bounds}") 
    else:
        print(f"Parameter {param} with value {calibrated} is within the bounds {bounds}")

n_histories = int(input("Enter the number of historical trajectories to generate for NPE data generation: "))
priors_bounds = torch.tensor(priors_bounds, dtype=torch.float32)
priors = BoxUniform(low=priors_bounds[:, 0], high=priors_bounds[:, 1])

n_stats = 6
raw = np.ndarray(shape = (n_histories, T_hist + 1))
npe_x = np.ndarray(shape = (n_histories, n_stats))

if "prior_check" in argv:
    plt.figure(figsize=(12, 8))

priors_samples = priors.sample((n_histories, ))
for i, draw in enumerate(priors_samples):
    sim_parameters = rep_parameters(hist_params, parameters_to_calibrate, draw)
    sim_initial = hist_initial
    sim_out = run_monte_carlo(sim_parameters, sim_initial, T_hist, num_simulations=15)
    if "prior_check" in argv:
        plt.plot(sim_out, alpha=0.2, color="blue")
    sim_stats = compute_statistics(sim_out)
    npe_x[i, :] = sim_stats
    raw[i, :] = sim_out
    print(f"Completed {i+1}/{n_histories} simulations for NPE data generation", end="\r")
print()

### saving
np.savez("npe_data.npz", priors_samples=priors_samples, npe_x=npe_x, raw=raw)

if "prior_check" in argv:
    plt.title("Simulated trajectories from the prior")
    plt.xlabel("Time (quarters)")
    plt.ylabel("Real GDP")
    plt.savefig(f"pngs/prior_check_n{n_histories}_p{len(parameters_to_calibrate)}.png")
    plt.show()
    plt.clf()

### NPE training
npe = NPE(prior=priors)
npe = npe.append_simulations(priors_samples, torch.tensor(npe_x, dtype=torch.float32))
npe.train()

posterior = npe.build_posterior()
print()

observed_series = df[df["date"] <= final_calibration]
realized_future = df[df["date"] > final_calibration]

assert len(observed_series) == T_hist + 1
assert len(realized_future) == T_forecast

forecast_statistic = torch.tensor(compute_statistics(observed_series["real"].values))

if "posterior_check" in argv:
    pass

if "sbc" in argv:
    num_sbc_samples = 10
    num_sbc_samples = 10
    thetas = priors.sample((num_sbc_samples,))
    params = [rep_parameters(final_params, parameters_to_calibrate, theta) for theta in thetas]
    xs = [run_monte_carlo(param, final_initial, T_forecast, num_simulations=10) for param in params]
    stats = [compute_statistics(x) for x in xs]
    stats = torch.tensor(stats, dtype=torch.float32)
    sbc_results = run_sbc(
        posterior=posterior,
        thetas=thetas,
        xs = stats,
        num_posterior_samples=100
    )

if "pairplot" in argv:

    _ = pairplot(
        posterior.sample((1000,), x = torch.tensor(forecast_statistic, dtype=torch.float32)),
        points=np.array([final_params[param] for param in parameters_to_calibrate])[None, :],
        labels=parameters_to_calibrate,
        title = "Pairplot of the posterior samples with calibrated values"
    )
    plt.savefig(f"pngs/pairplot_n{n_histories}_p{len(parameters_to_calibrate)}.png")
    plt.show()
    plt.clf()

if "post_check" in argv:
    samples = posterior.sample((100, ), x = torch.tensor(npe_x, dtype=torch.float32))
    plt.figure(figsize=(12, 8))
    for i, sample in enumerate(samples):
        sim_parameters = rep_parameters(hist_params, parameters_to_calibrate, sample)
        sim_initial = hist_initial
        sim_out = run_monte_carlo(sim_parameters, sim_initial, T_hist, num_simulations=15)
        plt.plot(sim_out, alpha=0.2, color="green")
    plt.title("Simulated trajectories from the posterior")
    plt.xlabel("Time (quarters)")
    plt.ylabel("Real GDP")
    plt.savefig(f"pngs/posterior_check_n{n_histories}_p{len(parameters_to_calibrate)}.png")
    plt.show()
    plt.clf()


abm_forecast = run_monte_carlo(final_params, final_initial, T_forecast, num_simulations=50)[1:]

npe_forecast_params = posterior.sample((1000, ), x = forecast_statistic).mean(dim= 0)
npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
npe_forecast = run_monte_carlo(npe_forecast_params, final_initial, T_forecast, num_simulations=50)[1:]

RMSFE_abm = np.sqrt(np.mean((abm_forecast - realized_future["real"].values)**2))
RMSFE_npe = np.sqrt(np.mean((npe_forecast - realized_future["real"].values)**2))

print(f"RMSFE of ABM forecast: {RMSFE_abm}")
print(f"RMSFE of NPE forecast: {RMSFE_npe}")

plt.figure(figsize=(12, 8))
plt.plot(range(1, T_forecast + 1), realized_future["real"], label="Real GDP", color="black")
plt.plot(range(1, T_forecast + 1), abm_forecast, label="ABM Forecast", color="blue")
plt.plot(range(1, T_forecast + 1), npe_forecast, label="NPE Forecast", color="orange")
plt.xticks(range(1, T_forecast + 1), realized_future["date"].dt.strftime("%Y-%m").values, rotation=45)
plt.title("Forecast comparison")
plt.xlabel("Time (quarters)")
plt.ylabel("Real GDP")
plt.legend()
plt.savefig(f"pngs/forecast_comparison_n{n_histories}_p{len(parameters_to_calibrate)}.png")
plt.show()
