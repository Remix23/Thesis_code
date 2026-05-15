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
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import (
    CausalCNNEmbedding
)

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

to_growth_rate = lambda x: np.diff(np.log(x))

STATISTICS = {
    "mean": lambda sim_out: np.mean(sim_out),
    "std": lambda sim_out: np.std(sim_out),
    "yearly_corr": lambda sim_out: np.corrcoef(sim_out[:-4], sim_out[4:])[0, 1],
    "ar1_coeff": lambda sim_out: findar_p(sim_out, 1)[1],
    "ar2_coeff": lambda sim_out: findar_p(sim_out, 2)[2],
    "min": lambda sim_out: np.min(sim_out),
    "max": lambda sim_out: np.max(sim_out),
    "skewness": lambda sim_out: np.mean((sim_out - np.mean(sim_out))**3) / np.std(sim_out)**3,
    "kurtosis": lambda sim_out: np.mean((sim_out - np.mean(sim_out))**4) / np.std(sim_out)**4,
    "quantile_25" : lambda sim_out: np.quantile(sim_out, .25),
    "quantile_50" : lambda sim_out: np.quantile(sim_out, .50),
    "quantile_75" : lambda sim_out: np.quantile(sim_out, .75),
    "recession_count" : lambda sim_out: np.sum(sim_out < 0),
}

short_names = {
    "mean": "m",
    "std": "s",
    "yearly_corr": "y_c",
    "ar1_coeff": "ar1",
    "ar2_coeff": "ar2",
    "min": "min_gr",
    "max": "max_gr",
    "skewness": "skew_gr",
    "kurtosis": "kurt_gr",
    "quantile_25": "q25_gr",
    "quantile_50": "q50_gr",
    "quantile_75": "q75_gr",
    "recession_count": "rec_c"
}

to_short_names = lambda keys: [short_names[key] for key in keys]

def compute_statistics (sim_out):
    # compute summary statistics from the simulated data
    # for example, we can compute the mean and standard deviation of real GDP
    growth_rates = to_growth_rate(sim_out)

    mean_gdp = np.mean(sim_out)
    mean_gdprowth = np.mean(growth_rates)

    std_gdp = np.std(sim_out)
    std_gdprowth = np.std(growth_rates)

    ### yearly correlation y_t and y_t-4
    yearly_corr = np.corrcoef(growth_rates[:-4], growth_rates[4:])[0, 1]
    ### AR(1) coefficient of real GDP
    ar1_coeff = findar_p(growth_rates, 1)[1]
    ar2_coeff = findar_p(growth_rates, 2)[2]

    mini, maxi = np.min(growth_rates), np.max(growth_rates)
    
    skewness_gdp = np.mean((growth_rates - mean_gdprowth)**3) / std_gdprowth**3
    return np.array([mean_gdprowth, std_gdprowth, yearly_corr, ar1_coeff, ar2_coeff, mini, maxi, skewness_gdp])

def compute_statistics_dict (sim_out, keys = None):
    x = to_growth_rate(sim_out)
    if keys is None:
        keys = STATISTICS.keys()
    return np.array([STATISTICS[key](x) for key in keys])

### load data
def load_data(file_path):
    data = np.load(file_path)
    required_keys = ["priors_samples", "raw", "parameters_to_calibrate", "bounds"]
    print(f"Loaded data from {file_path} with keys: {list(data.files)}")
    if not all(key in data.files for key in required_keys):
        raise ValueError(f"Data file must contain the following keys: {required_keys}")
    if not "npe_x" in data.files:
        print("Warning: 'npe_x' not found in data file. It will be set to None.")
        data["npe_x"] = None
    print("Parameters: ")
    for param, value in zip(data["parameters_to_calibrate"], data["bounds"]):
        print(f"{param}: {value}")

    return data["priors_samples"], data["npe_x"], data["raw"], data["parameters_to_calibrate"], data["bounds"]

data_folder = path.join(getcwd(), "data")
all_npe = listdir(data_folder) + listdir(getcwd())

files = [f for f in all_npe if f.endswith(".npz")]
if not files:
    raise FileNotFoundError("No .npz files found in the data folder or current directory.")

print("Available .npz files:")
for i, file in enumerate(files):
    print(f"{i + 1}. {file} | files: {np.load(path.join(data_folder, file)).files if file in listdir(data_folder) else np.load(path.join(getcwd(), file)).files}")

file_index = int(input("Enter the number corresponding to the file you want to load: ")) - 1

if file_index < 0 or file_index >= len(files):
    raise ValueError("Invalid file number selected.")

file_path = path.join(data_folder, files[file_index]) if files[file_index] in listdir(data_folder) else path.join(getcwd(), files[file_index])
priors_samples, npe_x_base, raw, parameters_to_calibrate, bounds = load_data(file_path)
parameters_to_calibrate = parameters_to_calibrate.tolist()

### initial calibration
initial_calibration = datetime(2010, 3, 31)

n = 6 # number of years to simulate -> 5 years
T_hist = 4 * n - 1 # quaters
final_calibration = datetime(initial_calibration.year + T_hist // 4, 12, initial_calibration.day)

T_forecast = 4 * 4 # 4 years forecast
date_forecast = datetime(final_calibration.year + T_forecast // 4, final_calibration.month, final_calibration.day)
### real data
csv_data = pd.read_csv("data/italy_real_gdp.csv", parse_dates=["observation_date"])
csv_data = csv_data[csv_data["observation_date"] >= initial_calibration]
quarterly_dates, bit = [np.array(x) for x in jl.get_real()]

n = len(quarterly_dates)
df = pd.DataFrame({"date": quarterly_dates.flatten(), "real": bit.flatten()})
df = df[df["date"] >= initial_calibration]
df = df[df["date"] <= date_forecast]

observed_series = df[df["date"] <= final_calibration]
realized_future = df[df["date"] > final_calibration]

assert len(observed_series) == T_hist + 1
assert len(realized_future) == T_forecast

hist_params, hist_initial = jl.calibrate(initial_calibration.year, initial_calibration.month, initial_calibration.day)

final_params, final_initial = jl.calibrate(final_calibration.year, final_calibration.month, final_calibration.day)

n_hist = priors_samples.shape[0]

priors = BoxUniform(low=torch.tensor(bounds)[:, 0], high=torch.tensor(bounds)[:, 1])

def train_npe_statistics (priors, prior_samples, raw, stat_keys):
    npe = NPE(prior=priors)
    npe_x = np.apply_along_axis(lambda sim_out: compute_statistics_dict(sim_out, stat_keys), 1, raw)
    npe_x = torch.tensor(npe_x, dtype=torch.float32)
    samples = torch.tensor(prior_samples, dtype=torch.float32)
    net = npe.append_simulations(samples, npe_x).train()
    posterior = npe.build_posterior()
    print()
    return posterior

def train_npe_nn (priors, prior_samples, raw, nn):
    dens_estimator = posterior_nn(
        model="maf",
        embedding_net=nn,
    )
    npe = NPE(prior=priors, density_estimator=dens_estimator)
    npe_x = torch.tensor(raw, dtype=torch.float32)
    samples = torch.tensor(prior_samples, dtype=torch.float32)
    net = npe.append_simulations(samples, npe_x).train()
    posterior = npe.build_posterior()
    return posterior

### NPE training
if npe_x_base is not None:
    npe = NPE(prior=priors)
    npe = npe.append_simulations(torch.tensor(priors_samples, dtype=torch.float32), torch.tensor(npe_x_base, dtype=torch.float32))
    npe.train()

    posterior = npe.build_posterior()
    print()

def pplot_stat (posterior, n_histories, observed_series, validation_series, final_params, stat_keys):
    forecast_statistics = torch.tensor(compute_statistics_dict(observed_series, stat_keys), dtype=torch.float32)
    posterior_samples = posterior.sample((1000,), x = forecast_statistics)
    _ = pairplot(
        posterior_samples,
        points=np.array([final_params[param] for param in parameters_to_calibrate])[None, :],
        labels=parameters_to_calibrate,
        limits = torch.tensor(bounds, dtype=torch.float32),
        title = "Pairplot of the posterior samples with calibrated values"
    )
    plt.show()
    plt.savefig(f"pngs/pairplot_n{n_histories}_p{len(parameters_to_calibrate)}_stats_{', '.join(stat_keys)}.png")

### check later for nn embeddings
def forecast (posterior, observed_series, final_params, final_initial, stat_keys = None):
    if stat_keys is not None:
        forecast_statistics = torch.tensor(compute_statistics_dict(observed_series, stat_keys), dtype=torch.float32)
        posterior_samples = posterior.sample((1000,), x = forecast_statistics)
    else:
        posterior_samples = posterior.sample((1000, ), x = observed_series)
    npe_forecast_params = posterior_samples.mean(dim= 0)
    npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
    npe_forecast = run_monte_carlo(npe_forecast_params, final_initial, T_forecast, num_simulations=100)[1:]
    return npe_forecast

def plot_forecasts (forecasts : dict[str, np.ndarray], realized_future, parameters_to_calibrate, bounds):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, T_forecast + 1), realized_future["real"], label="Real GDP", color="black")
    rolling_rsmfes = {}
    for key, forecast in forecasts.items():
        rmsfe = compute_rmsfes(forecast, realized_future["real"].values)
        rolling_rsmfes[key] = rmsfe
        print(f"RMSFE of {key} forecast: {rmsfe[-1]:.4f}")
        plt.plot(range(1, T_forecast + 1), forecast, label=f"{key} Forecast")
    plt.xticks(range(1, T_forecast + 1), realized_future["date"].dt.strftime("%Y-%m").values, rotation=45)
    plt.title("Forecast comparison")
    plt.xlabel("Time (quarters)")
    plt.ylabel("Real GDP")
    plt.legend()
    plt.savefig(f"pngs/forecast_comparison_n{n_hist}_p{', '.join(parameters_to_calibrate)}_bounds{', '.join(str(b) for b in bounds)}.png")
    plt.show()
    return rolling_rsmfes

def compute_rmsfes (forecast, realized):
    rolling_differences = [(forecast[:i] - realized[:i])**2 for i in range(1, len(forecast) + 1)]
    means = np.array(list(map(np.mean, rolling_differences)))
    return np.sqrt(means)

### checks
s1 = ["mean", "std", "yearly_corr", "ar1_coeff", "min", "max", "skewness"]
s1_prime = ["mean", "std", "yearly_corr", "ar1_coeff", "min", "max", "skewness", "kurtosis", "recession_count"]
s2 = ["mean", "std", "yearly_corr", "ar1_coeff", "ar2_coeff", "min", "max", "skewness", "kurtosis"]
s3 = ["mean", "std", "yearly_corr", "ar1_coeff", "skewness", "kurtosis", "quantile_25", "quantile_50", "quantile_75", "recession_count"]

versions = [s1]

ps = [train_npe_statistics(priors, priors_samples, raw, s) for s in versions]
forecasts = {",".join(to_short_names(s)): forecast(p, observed_series["real"].values, final_params, final_initial, s) for p, s in zip(ps, versions)}

abm_forecast = run_monte_carlo(final_params, final_initial, T_forecast, num_simulations=100)[1:]
forecasts["ABM_base"] = abm_forecast

if npe_x_base is not None:
    forecast_statistic = torch.tensor(compute_statistics(observed_series["real"].values))
    npe_forecast_params = posterior.sample((1000, ), x = forecast_statistic).mean(dim= 0)
    npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
    npe_forecast = run_monte_carlo(npe_forecast_params, final_initial, T_forecast, num_simulations=100)[1:]

if npe_x_base is not None:
    forecasts["NPE_base"] = npe_forecast

rolling_rsmfes = plot_forecasts(forecasts, realized_future, parameters_to_calibrate, bounds)
