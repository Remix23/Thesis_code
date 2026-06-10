from os import path, listdir, getcwd, environ
from sys import argv
from datetime import datetime, date
from typing import Tuple
NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl
jl.seval("using JuliaModel: run_simulation, get_real, calibrate, run_monte_carlo")
print(f"Using JuliaModel with {jl.seval('Threads.nthreads()')} threads.")

from sbi.utils import BoxUniform
from sbi.inference import NPE, NPE_C, simulate_for_sbi, NRE
from sbi.analysis import pairplot
from sbi.neural_nets import posterior_nn, classifier_nn
from sbi.neural_nets.embedding_nets import (
    CausalCNNEmbedding
)

from neural_network import RNN, CNN_GDP, SeqEmbedding, Batched

from sbi.diagnostics import check_sbc, run_sbc
from sbi.analysis import sbc_rank_plot

import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import pickle

torch.manual_seed(0)
np.random.seed(0)

NUM_CALIBRATION_DATES = 5
NUM_SIM_PER_ROUND = 1000

NUM_RUNS_PER_DRAW = 5

ROUNDS = 10

### num of simulations per round: NUM_SIM_PER_ROUND * NUM_CALIBRATION_DATES * NUM_RUNS_PER_DRAW for batched
### num of simulations per round for non-batched: NUM_SIM_PER_ROUND * NUM_RUNS_PER_DRAW

FIRST_CALIBRATION_DATE = datetime(2011, 3, 31)

AVAILABLE_KEYS = [
    "real_gdp",
    "nominal_gdp",
    "real_gva",
    "nominal_gva",
    "gdp_deflator",
    "real_household_consumption", 
    "real_government_consumption",
    "real_capitalformation",
    "real_exports",
    "real_imports",
    "wages",
    "euribor",
    "gdp_deflator",
]

def rep_parameters(parameters, parameters_to_calibrate, draw):
    sim_paramerters = parameters.copy()
    for param, value in zip(parameters_to_calibrate, draw.tolist()):
        sim_paramerters[param] = value
    return sim_paramerters

def run_sim(parameters, initial_conditions, T, keys = ["real_gdp"]):
    data = jl.run_simulation(parameters, initial_conditions, T, keys)
    return np.array(data)

def run_monte_carlo(parameters, initial_conditions, T, num_simulations, keys = ["real_gdp"]):
    data = jl.run_monte_carlo(parameters, initial_conditions, T, num_simulations, keys)
    data = np.mean(data, axis=0) # average across simulations
    return np.array(data).flatten()

def findar_p (times_series, p):
    target = times_series[p:]
    lagged = np.array([times_series[i:-(p - i)] for i in range(p)]).T
    ones = np.ones((lagged.shape[0], 1))
    lagged = np.hstack((ones, lagged))
    ols = np.linalg.lstsq(lagged, target, rcond=None)
    return ols[0]

def gen_sample (calibration_date, theta_draws, T, params_to_calibrate, n_runs, keys):
    params, initial_conditions = jl.calibrate(
        calibration_date.year, calibration_date.month, calibration_date.day
    )
    samples = torch.zeros((theta_draws.shape[0], n_runs, len(keys), T + 1))
    for i, draw in enumerate(theta_draws):
        sim_parameters = rep_parameters(params, params_to_calibrate, draw)
        sim_out = np.array(jl.run_monte_carlo(sim_parameters, initial_conditions, T, n_runs, keys))
        sim_out = torch.tensor(sim_out, dtype=torch.float32)
        samples[i, :, :, :] = sim_out

    return samples

def gen_batch (calibration_date, num_calibrations, T, priors, params_to_calibrate, n_samples, n_runs, keys) -> Tuple[torch.Tensor, torch.Tensor]:

    batch = torch.zeros(
        n_samples, num_calibrations, n_runs, len(keys), T + 1
    )
    if isinstance(priors, BoxUniform):
        theta_draws = priors.sample((n_samples, ))
    else:
        theta_draws = priors.sample((n_samples, ), show_progress_bars=False)

    for i in range(num_calibrations):
        cal_year = calibration_date.year + i
        cal_date = datetime(cal_year, calibration_date.month, calibration_date.day)
        print(f"Generating batch for calibration date: {cal_date.strftime('%Y-%m-%d')}")
        samples = gen_sample(cal_date, theta_draws, T, params_to_calibrate, n_runs, keys)
        batch[:, i, :, :, :] = samples

    return batch, theta_draws

def prepare_real (observed, keys, num_calibrations):
    ### we want observed to a dataframe with columns "date" and keys, and sorted by date, and with at least T_hist + 1 rows for each calibration date, and we want to return a generator of tensors of shape (len(keys), T_hist + 1) for each calibration date, where the tensor is the historical data for that calibration date.
    ### we output a x_o tensor for NN
    ### we need (calibration dates, n_runs, n_feature, T_hist + 1)

    out = torch.zeros((num_calibrations, NUM_RUNS_PER_DRAW, len(keys), T_hist + 1))

    for i in range(num_calibrations):
        cal_year = FIRST_CALIBRATION_DATE.year + i
        cal_date = datetime(cal_year, FIRST_CALIBRATION_DATE.month, FIRST_CALIBRATION_DATE.day)
        print(f"Preparing real data for calibration date: {cal_date.strftime('%Y-%m-%d')}")
        observed_cal = observed[observed["date"] >= cal_date]
        if len(observed_cal) < T_hist + 1:
            raise ValueError(f"Not enough historical data for calibration date {cal_date.strftime('%Y-%m-%d')}. Required: {T_hist + 1}, Available: {len(observed_cal)}")
        observed_cal = observed_cal.tail(T_hist + 1)
        
        for j in range(NUM_RUNS_PER_DRAW):
            out[i, j, :, :] = torch.tensor(observed_cal[keys].values.T, dtype=torch.float32)

    return out

to_log_growth = lambda x: np.diff(np.log(x), axis=-1)


to_log = lambda x: torch.log(x)
to_growth_rate = lambda x: torch.diff(x, dim=-1) / x[..., :-1]

### x - (n_trajectories, T_hist) -> (n_trajectories, (T_hist - 1)*3) with channels log, growth rate, difference
# we want 
to_cnn3 = lambda x: torch.concatenate([to_log(x)[:, 1:], to_growth_rate(x), torch.diff(x, dim=-1)], dim=-1)
# to_cnn3_channels = lambda x: np.concatenate([to_log(x)[:, 1:], to_growth_rate(x), np.diff(x)], axis=1)

### input transormations for NN-based NPEs
# x: (N_draws, N_calibration_dates, n_runs, n_features, T_hist + 1)
reduction = lambda x: x.mean(axis=2)[:, 0, 0, :] 
to_one_series = lambda x: x[:, 0, 0, 0, :] # take the first calibration date, first run, first feature
to_seq_multivariate = lambda x: x.mean(axis=2)[:, 0, :, 1:] # take the first calibration date, average across runs, keep all features

STATISTICS = {
    "mean": lambda sim_out: np.mean(sim_out),
    "std": lambda sim_out: np.std(sim_out),
    "yearly_corr": lambda sim_out: np.corrcoef(sim_out[:-4], sim_out[4:])[0, 1],
    "auto_corr_1" : lambda sim_out: np.corrcoef(sim_out[:-1], sim_out[1:])[0, 1],
    "auto_corr_2" : lambda sim_out: np.corrcoef(sim_out[:-2], sim_out[2:])[0, 1],
    "auto_corr_3" : lambda sim_out: np.corrcoef(sim_out[:-3], sim_out[3:])[0, 1],
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
    "sequential" : lambda sim_out: None,
}

short_names = {
    "mean": "m",
    "std": "s",
    "yearly_corr": "y_c",
    "auto_corr_1": "ac1",
    "auto_corr_2": "ac2",
    "auto_corr_3": "ac3",
    "ar1_coeff": "ar1",
    "ar2_coeff": "ar2",
    "min": "min_gr",
    "max": "max_gr",
    "skewness": "skew_gr",
    "kurtosis": "kurt_gr",
    "quantile_25": "q25_gr",
    "quantile_50": "q50_gr",
    "quantile_75": "q75_gr",
    "recession_count": "rec_c",
    "sequential" : "seq",
}

to_short_names = lambda keys: [short_names[key] for key in keys]

def compute_statistics (sim_out):
    # compute summary statistics from the simulated data
    # for example, we can compute the mean and standard deviation of real GDP
    growth_rates = to_log_growth(sim_out)

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
    x = to_log_growth(sim_out)
    if keys is None:
        keys = STATISTICS.keys()
    return np.array([STATISTICS[key](x) for key in keys if key != "sequential"])

### load data
def load_custom ():
    parameters_to_calibrate = ["theta", "zeta", "zeta_LTV", "zeta_b", "omega", "lambda_p"]

    priors_bounds = [
        (0, 1),  # theta
        (0.01, 0.5),  # zeta
        (0.3, 1),  # zeta_LTV
        (0, 2),  # zeta_B
        (0, 1),  # omega
        (0.1, 4),  # lambda_p
    ]
    return np.zeros((1, 1)), None, None, parameters_to_calibrate, priors_bounds


def load_data():
    data_folder = path.join(getcwd(), "data_npz")
    all_npe = listdir(data_folder) + listdir(getcwd())

    files = [f for f in all_npe if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError("No .npz files found in the data folder or current directory.")

    print("Available .npz files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file} | files: {np.load(path.join(data_folder, file)).files if file in listdir(data_folder) else np.load(path.join(getcwd(), file)).files}")

    file_index = int(input("Enter the number corresponding to the file you want to load: ")) - 1


    if file_index < 0 or file_index >= len(files):
        return load_custom()

    file_path = path.join(data_folder, files[file_index]) if files[file_index] in listdir(data_folder) else path.join(getcwd(), files[file_index])

    data = np.load(file_path)
    required_keys = ["priors_samples", "raw", "parameters_to_calibrate", "bounds"]
    print(f"Loaded data from {file_path} with keys: {list(data.files)}")
    if not all(key in data.files for key in required_keys):
        raise ValueError(f"Data file must contain the following keys: {required_keys}")
    if not "npe_x" in data.files:
        print("Warning: 'npe_x' not found in data file. It will be set to None.")
        dat = None
    else:
        dat = data["npe_x"]
    print("Parameters: ")
    for param, value in zip(data["parameters_to_calibrate"], data["bounds"]):
        print(f"{param}: {value}")
    return data["priors_samples"], dat, data["raw"], data["parameters_to_calibrate"], data["bounds"]


def save_posteriors (posteriors, verions, file_name):
    with open(file_name, "wb") as f:
        pickle.dump({"posteriors": posteriors, "statistics_versions": verions}, f)
    print(f"Saved posteriors to {file_name}")

def remove_outliers(data: np.ndarray, m: float = 2.0, stat_names: list = None) -> np.ndarray:
    ### data is suppoer to 2dim, returns a boolean mask of the same shape as data, where True indicates that the corresponding value in data is not an outlier, and False indicates that it is an outlier.
    q3 = np.percentile(data, 75, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - m * iqr
    upper_bound = q3 + m * iqr
    out = np.any((data < lower_bound) | (data > upper_bound), axis=1)
    num_of_outliers = np.sum(out)
    print(f"Removing {num_of_outliers} outliers from the data [from {data.shape[0]} samples]")
    for stat, out_counts in enumerate(np.sum((data < lower_bound) | (data > upper_bound), axis=0)):
        if stat_names is not None:
            print(f"  - {to_short_names(stat_names)[stat]}: {out_counts} outliers")
        else:
            print(f"  - {stat}: {out_counts} outliers")
    return ~out

def train_npe_nn_batched (priors, nn, nn_transform, params_to_calibrate, observed, t_train, features_kyes, rounds = 3, n_sim_per_round = 100, num_calibrations = NUM_CALIBRATION_DATES):
    dense_estimator = posterior_nn(
        model="nsf",
        embedding_net=nn,
        z_score_x="structured"
    )
    x_o = nn_transform(prepare_real(observed, features_kyes, num_calibrations).unsqueeze(0)).flatten() # (1, n_features, T_hist + 1) -> (1, embedding_dim)
    inference = NPE_C(prior = priors, density_estimator=dense_estimator)
    proposal = priors
    proposals = []
    for i in range(rounds):

        batch, theta_draws = gen_batch(FIRST_CALIBRATION_DATE, num_calibrations, 
                                       t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes)
        
        # print(f"Generated batch with shape {batch.shape} for round {i + 1}/{rounds}")
        batch = nn_transform(batch)
        # print(f"Transformed batch shape after NN transform: {batch.shape}")
        batch = batch.flatten(start_dim=1)

        density_estimator = inference.append_simulations(theta_draws, batch, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with NN embedding {nn.__class__.__name__}")
    return posterior, proposals

def train_nre_nn_batched (priors, nn, nn_transform, params_to_calibrate, observed, t_train, features_kyes, rounds = 3, n_sim_per_round = 100, num_calibrations = NUM_CALIBRATION_DATES):
    classifier = classifier_nn(
        model="mlp",
        embedding_net_x=nn,
        z_score_x="structured"
    )
    x_o = nn_transform(prepare_real(observed, features_kyes, num_calibrations).unsqueeze(0)).flatten()
    inference = NRE(prior = priors, classifier=classifier)
    proposal = priors
    proposals = []
    for i in range(rounds):
        proposals.append(proposal)

        batch, theta_draws = gen_batch(FIRST_CALIBRATION_DATE, num_calibrations, 
                                       t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes)
        batch = nn_transform(batch)
        batch = batch.flatten(start_dim=1)
        
        classifier = inference.append_simulations(theta_draws, batch).train()
        posterior = inference.build_posterior(density_estimator=classifier)
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NRE training with NN embedding {nn.__class__.__name__}")
    return posterior, proposals

def train_npe_statistics_rounds (priors, stat_keys, full_params, initial_conditions, params_to_calibrate,observed, t_train, rounds = 3, n_sim_per_round = 100, features_kyes = None):
    
    inference = NPE_C(prior = priors)
    x_o = compute_statistics_dict(observed, stat_keys)
    x_o = torch.tensor(x_o, dtype=torch.float32)
    proposal = priors

    if features_kyes is None:
        features_kyes = ["real_gdp"]

    proposals = []
    for i in range(rounds):
        proposals.append(proposal)
        if i > 0:
            theta = proposal.sample((n_sim_per_round, ), show_progress_bars=False)
        else:
            theta = proposal.sample((n_sim_per_round, ))
        sim_data = np.array([run_monte_carlo(rep_parameters(full_params, params_to_calibrate, sample.numpy()), initial_conditions, t_train, num_simulations=1) for sample in theta])
        x = np.array([compute_statistics_dict(sim_out, stat_keys) for sim_out in sim_data])
        idx = remove_outliers(x, m = 8, stat_names=stat_keys)
        theta = theta[idx, :]
        x = x[idx, :]
        x = torch.tensor(x, dtype=torch.float32)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator=density_estimator, sample_with="mcmc")
        proposal = posterior.set_default_x(x_o)
        
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with statistics {', '.join(stat_keys)}")

    return posterior, proposals

def train_npe_nn_rounds (priors, nn, nn_transform, full_params, initial_conditions, params_to_calibrate,observed, t_train, rounds = 3, n_sim_per_round = 100, features_kyes = None):
    dense_estimator = posterior_nn(
        model="nsf",
        embedding_net=nn,
        z_score_x="structured"
    )

    if features_kyes is None:
        features_kyes = ["real_gdp"]

    x_o = torch.tensor(nn_transform(observed), dtype=torch.float32)
    inference = NPE_C(prior = priors, density_estimator=dense_estimator)
    proposal = priors
    proposals = []
    for i in range(rounds):
        proposals.append(proposal)

        if i > 0:
            theta = proposal.sample((n_sim_per_round, ), show_progress_bars=False)
        else:
            theta = proposal.sample((n_sim_per_round, ))

        # batch, theta_draws = gen_batch(FIRST_CALIBRATION_DATE, 
        #                                t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes)

        paramters = [rep_parameters(full_params, params_to_calibrate, sample.numpy()) for sample in theta]
        sim_data = np.array([run_monte_carlo(params, initial_conditions, t_train, num_simulations=1) for params in paramters])
        sim_data = sim_data.reshape((sim_data.shape[0], -1)) # reshape to (n_simulations, T_hist + 1) for CNN input
        
        ### batch: (n_samples, NUM_CALIBRATION_DATES, n_runs, len(features_kyes), T_hist + 1) -> (n_samples * NUM_CALIBRATION_DATES * n_runs * len(features_kyes), T_hist + 1)
        x = nn_transform(sim_data)
        x = torch.tensor(x, dtype=torch.float32)
        
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator=density_estimator, sample_with="mcmc")
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with NN embedding {nn.__class__.__name__}")
    return posterior, proposals

def train_nre_nn_rounds (priors, nn, nn_transform, full_params, initial_conditions, params_to_calibrate,observed, t_train, rounds = 3, n_sim_per_round = 100, features_kyes = None):
    classifier = classifier_nn(
        model="mlp",
        embedding_net_x=nn,
        z_score_x="structured"
    )

    if features_kyes is None:
        features_kyes = ["real_gdp"]

    x_o = torch.tensor(nn_transform(observed), dtype=torch.float32)
    inference = NRE(prior = priors, classifier=classifier)
    proposal = priors
    proposals = []

    for i in range(rounds):
        proposals.append(proposal)

        if i > 0:
            theta = proposal.sample((n_sim_per_round, ), show_progress_bars=False)
        else:
            theta = proposal.sample((n_sim_per_round, ))

        paramters = [rep_parameters(full_params, params_to_calibrate, sample.numpy()) for sample in theta]
        sim_data = np.array([run_monte_carlo(params, initial_conditions, t_train, num_simulations=NUM_RUNS_PER_DRAW) for params in paramters])
        sim_data = sim_data.reshape((sim_data.shape[0], -1)) # reshape to (n_simulations, T_hist + 1) for CNN input
        
        x = nn_transform(sim_data)
        x = torch.tensor(x, dtype=torch.float32)
        
        classifier = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator=classifier)
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NRE training with NN embedding {nn.__class__.__name__}")
    return posterior, proposals

def train_nre_statistics_rounds (priors, stat_keys, full_params, initial_conditions, params_to_calibrate, observed, t_train, rounds = 3, n_sim_per_round = 100, features_kyes = None):
    inference = NRE(prior = priors)
    x_o = compute_statistics_dict(observed, stat_keys)
    x_o = torch.tensor(x_o, dtype=torch.float32)
    proposal = priors

    if features_kyes is None:
        features_kyes = ["real_gdp"]

    proposals = []
    for i in range(rounds):
        proposals.append(proposal)
        if i > 0:
            theta = proposal.sample((n_sim_per_round, ), show_progress_bars=False)
        else:
            theta = proposal.sample((n_sim_per_round, ))
        sim_data = np.array([run_monte_carlo(rep_parameters(full_params, params_to_calibrate, sample.numpy()), initial_conditions, t_train, num_simulations=NUM_RUNS_PER_DRAW) for sample in theta])
        x = np.array([compute_statistics_dict(sim_out, stat_keys) for sim_out in sim_data])
        idx = remove_outliers(x, m = 8, stat_names=stat_keys)
        theta = theta[idx, :]
        x = x[idx, :]
        x = torch.tensor(x, dtype=torch.float32)

        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        
        print(f"\nCompleted round {i + 1}/{rounds} of NRE training with statistics {', '.join(stat_keys)}")

    return posterior, proposals

def pplot_stat (posterior, params, filename):

    posterior_samples = posterior.sample((1000,))
    _ = pairplot(
        posterior_samples,
        points=np.array([params[param] for param in parameters_to_calibrate])[None, :],
        labels=parameters_to_calibrate,
        limits = torch.tensor(bounds, dtype=torch.float32),
        title = "Pairplot of the posterior samples with calibrated values"
    )

    plt.savefig(f"pngs/pairplot_r{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_{filename}.png")
    plt.close()

def ppc_plot (observed_values, sim_out_transformed, filename):
    fig, axs = plt.subplots(1, len(observed_values), figsize=(24, 6))
    if len(observed_values) == 1:
        axs = [axs]
    for i, val in enumerate(observed_values):
        axs[i].hist(sim_out_transformed[:, i], bins=25, alpha=0.7)
        axs[i].axvline(val, color="red", linestyle="dashed", linewidth=2)
        axs[i].set_title(f"Statistic {i}")
    plt.suptitle("Posterior Predictive Check")
    plt.savefig(filename)
    plt.close()

def ppc_trajectories (observed_trajectory, sim_out_trajectories, filename):
    plt.figure(figsize=(12, 8))
    for traj in sim_out_trajectories:
        plt.plot(traj, color="blue", alpha=0.1)
    plt.plot(observed_trajectory, color="red", linewidth=2, label="Observed Trajectory")
    plt.title("Posterior Predictive Check: Trajectories")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def forecast (posterior, final_params, final_initial):

    posterior_samples = posterior.sample((1000, ))
    npe_forecast_params = posterior_samples.mean(dim= 0)
    npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
    npe_forecast = run_monte_carlo(npe_forecast_params, final_initial, T_forecast, num_simulations=100)[1:]
    return npe_forecast

def plot_forecasts (forecasts : dict[str, np.ndarray], realized_future, parameters_to_calibrate, bounds):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, T_forecast + 1), realized_future["real_gdp"], label="Real GDP", color="black")
    rolling_rsmfes = {}
    for key, forecast in forecasts.items():
        rmsfe = compute_rmsfes(forecast, realized_future["real_gdp"].values)
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

if __name__ == "__main__":

    if "load" in argv:
         priors_samples, npe_x_base, raw, parameters_to_calibrate, bounds = load_data()
    else:
        priors_samples, npe_x_base, raw, parameters_to_calibrate, bounds = load_custom()
    # parameters_to_calibrate = parameters_to_calibrate.tolist()

    ### initial calibration
    initial_calibration = FIRST_CALIBRATION_DATE

    n = 3 # number of years to simulate -> 5 years
    T_hist = 4 * n - 1 # quaters
    final_calibration = datetime(initial_calibration.year + NUM_CALIBRATION_DATES, 12, initial_calibration.day)

    T_forecast = 4 * 3 # 4 years forecast
    date_forecast = datetime(final_calibration.year + T_forecast // 4, final_calibration.month, final_calibration.day)
    ### real data

    keys = [
        "real_gdp",
        "gdp_deflator",
        "nominal_gva",
    ]

    data, quarterly_dates = jl.get_real(keys)

    df = pd.DataFrame({"date": np.array(quarterly_dates).flatten(), **{key: np.array(data[key]).flatten() for key in keys}})

    df = df[df["date"] >= initial_calibration]

    observed_series = df[df["date"] <= final_calibration + pd.DateOffset(years=3)]

    realized_future = df[df["date"] > final_calibration]
    realized_future = realized_future.head(T_forecast)

    hist_params, hist_initial = jl.calibrate(initial_calibration.year, initial_calibration.month, initial_calibration.day)

    final_params, final_initial = jl.calibrate(final_calibration.year, final_calibration.month, final_calibration.day)

    hist_params["omega"] = 0.5
    hist_params["lambda_p"] = 2

    final_params["omega"] = 0.5
    final_params["lambda_p"] = 2

    n_hist = priors_samples.shape[0]

    priors = BoxUniform(low=torch.tensor(bounds)[:, 0], high=torch.tensor(bounds)[:, 1])

    ### NPE training
    if npe_x_base is not None and "base" in argv:
        npe = NPE(prior=priors)
        idx = remove_outliers(npe_x_base, m = 5)
        npe = npe.append_simulations(torch.tensor(priors_samples[idx, :], dtype=torch.float32), torch.tensor(npe_x_base[idx, :], dtype=torch.float32))
        npe.train()

        posterior = npe.build_posterior()
        print()

    ### statistics
    s1 = ["mean", "std", "yearly_corr", "ar1_coeff", "min", "max", "skewness", "quantile_50", "recession_count"]
    s1_prime = ["mean", "std", "yearly_corr", "ar1_coeff", "min", "skewness", "kurtosis", "recession_count"]
    s2 = ["mean", "std", "yearly_corr", "ar1_coeff", "ar2_coeff", "min", "skewness", "kurtosis"]
    s_base = ["mean", "std", "min", "max", "auto_corr_1", "auto_corr_2", "auto_corr_3","quantile_25", "quantile_50", "quantile_75"]

    s_base_seq = s_base + ["sequential"]
    s1_seq = s1 + ["sequential"]    

    ### NPE
    statistics_versions = [s_base_seq]
    statistics_posteriors = []
    statistics_nres = []
    posteriors_hist = []

    if not "load_posteriors" in argv:

        for i, s in enumerate(statistics_versions):
            final_posterior, posteriors = train_npe_statistics_rounds(priors, s, hist_params, hist_initial, 
                                            parameters_to_calibrate, observed_series["real_gdp"].values[:T_hist + 1], 
                                            t_train = T_hist, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND)
            posteriors_hist.append((s, posteriors))
            statistics_posteriors.append(final_posterior)
        
        for i, s in enumerate(statistics_versions):
            final_ratio, ratios = train_nre_statistics_rounds(priors, s, hist_params, hist_initial, 
                                            parameters_to_calibrate, observed_series["real_gdp"].values[:T_hist + 1], 
                                            t_train = T_hist, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND)
            
            statistics_nres.append(final_ratio)

    ### networking embedding
    nns = []
    nn_posteriors = []
    nn_nres = []
    nn_transforms = []
    nn_versions = []
    nn_keys = []
    nn_cal_nums = []

    if "nn" in argv:

        nn_raw = CausalCNNEmbedding(
            input_shape=(T_hist, ), # length of the time series
            num_conv_layers=2,
            pool_kernel_size=3,
            output_dim=12,
        )

        nn_diff = CausalCNNEmbedding(
            input_shape=(T_hist, ), ### because of the difference
            num_conv_layers=2,
            pool_kernel_size=3,
            output_dim=12,
        )

        # ### three channels - log, growth rate, difference (as detrended)
        nn_3channels = CausalCNNEmbedding(
            input_shape=(T_hist, ), ### because of the difference
            in_channels=3,
            num_conv_layers=2,
            pool_kernel_size=3,
            output_dim=12,
        )

        ### rnn as in Dyer et. al. (2024) and labour market
        nn_rnn = RNN(
            input_dim=T_hist,
            flavour="gru",
            hidden_dim=32,
            num_layers=2,
            mlp_dims=12
        )

        seq_multivariate = SeqEmbedding(
            T = T_hist,
            n_features = len(keys),
            hidden_size = 64,
            out_dim = 16
        )

        batched_multivariate = Batched(
            calibration_dates=NUM_CALIBRATION_DATES,
            n_runs=NUM_RUNS_PER_DRAW,
            n_features=len(keys),
            T=T_hist,
            hidden_size=64,
            out_dim=16
        )


        s_nn = ["mean", "std", "yearly_corr", "ar1_coeff", "ar2_coeff", "min", "skewness", "kurtosis"]

        nn_cnn_mixture = CNN_GDP(stat_keys=s_nn, in_channels=3, conv_dims=[16, 32], summary_dims=16, out_dims=32, pool_kernel_size=1)

        nns += [nn_raw, nn_diff, nn_3channels, nn_rnn, nn_cnn_mixture]
        nn_transforms += [lambda x: reduction(x)[:, 1:],lambda x: to_growth_rate(reduction(x)), lambda x: to_cnn3(reduction(x)), lambda x: reduction(x)[:, 1:], lambda x: to_cnn3(reduction(x))]
        # nn_transforms += [lambda x: reduction(x)[:, 1:], lambda x: to_growth_rate(reduction(x)), lambda x: x[..., 1:],  to_seq_multivariate]
        nn_versions += ["raw", "diff", "3channels", "rnn", "cnn_mixture"]
        nn_keys += [["real_gdp"], ["real_gdp"], ["real_gdp"], ["real_gdp"], ["real_gdp"]]
        nn_cal_nums += [1, 1, 1, 1, 1]

        assert len(nns) == len(nn_versions) == len(nn_transforms) == len(nn_keys), "Length of nns, nn_versions, nn_transforms and nn_keys must be the same"

    ### append NN-based posteriors
    if not "load_posteriors" in argv:
        print("Training NN-based NPEs...")
        for nn, transform, name, key_list, cal_num in zip(nns, nn_transforms, nn_versions, nn_keys, nn_cal_nums):
            print(f"\nTraining NPE with NN embedding: {name}")
            post, posteriors = train_npe_nn_batched(priors, nn, transform, parameters_to_calibrate, observed_series, T_hist, key_list, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND, num_calibrations=cal_num   )
            # post = train_npe_nn(priors, priors_samples, transform(raw), nn)
            nn_posteriors.append(post)

        print("Training NN-based NREs...")
        for nn, transform, name, key_list, cal_num in zip(nns, nn_transforms, nn_versions, nn_keys, nn_cal_nums):
            print(f"\nTraining NRE with NN embedding: {name}")
            post, posteriors = train_nre_nn_batched(priors, nn, transform, parameters_to_calibrate, observed_series, T_hist, key_list,  rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND, num_calibrations=cal_num)
            nn_nres.append(post)

    if "load_posteriors" in argv:
        post_dir = "trained_posteriors"
        post_files = [f for f in listdir(post_dir) if f.endswith(".pkl")]
        for i, file in enumerate(post_files):
            print(f"{i + 1}. {file}")
        file_index = int(input("Enter the number corresponding to the file you want to load: ")) - 1
        
        if file_index < 0 or file_index >= len(post_files):
            print("Invalid index. Exiting.")
            exit()
        base_name = post_files[file_index].split("_")[1:]
        posterior_name = "posteriors_" + "_".join(base_name)
        ratio_name = "ratios_" + "_".join(base_name)
        posteriors_path = path.join(post_dir, posterior_name)
        ratio_path = path.join(post_dir, ratio_name)
        with open(posteriors_path, "rb") as f:
            data = pickle.load(f)
            posteriors = data["posteriors"]

        with open(ratio_path, "rb") as f:
            data = pickle.load(f)
            nres = data["posteriors"]

        for key, posterior in posteriors.items():
            if key.startswith("stat_"):
                statistics_posteriors.append(posterior)
            elif key.startswith("nn_"):
                nn_posteriors.append(posterior)

        for key, nre in nres.items():
            if key.startswith("stat_"):
                statistics_nres.append(nre)
            elif key.startswith("nn_"):
                nn_nres.append(nre)
        print(f"[NPE] Loaded {len(statistics_posteriors)} statistics-based posteriors\n[NPE] Loaded {len(nn_posteriors)} NN-based posteriors from {posterior_name}")
        print(f"[NRE] Loaded {len(statistics_nres)} statistics-based NREs\n[NRE] Loaded {len(nn_nres)} NN-based NREs from {ratio_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "save" in argv:
        posteriors = {f"stat_{','.join(to_short_names(s))}": p for p, s in zip(statistics_posteriors, statistics_versions)}
        
        posteriors.update({f"nn_{name}": p for p, name in zip(nn_posteriors, nn_versions)})

        ratios = {f"stat_{','.join(to_short_names(s))}": r for r, s in zip(statistics_nres, statistics_versions)}
        ratios.update({f"nn_{name}": r for r, name in zip(nn_nres, nn_versions)})

        save_posteriors(posteriors, statistics_versions, f"trained_posteriors/posteriors_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_{timestamp}.pkl")
        save_posteriors(ratios, statistics_versions, f"trained_posteriors/ratios_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_{timestamp}.pkl")

    if "pp_history" in argv:
        ### pairplots for statistics-based NPE
        for p, s in zip(statistics_posteriors, statistics_versions):
            pplot_stat(p, hist_params, filename = "npe_ " + ",".join(to_short_names(s)) + "_hist")

        for r, s in zip(statistics_nres, statistics_versions):
            pplot_stat(r, hist_params, filename = "nre_ " + ",".join(to_short_names(s)) + "_hist")

        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            pplot_stat(p, hist_params, filename = f"nn_npe_{name}_hist")


        for r, transform, name in zip(nn_nres, nn_transforms, nn_versions):
            pplot_stat(r, hist_params, filename = f"nn_nre_{name}_hist")

    if "ppc" in argv:
        ### for statistics
        for p, s in zip(statistics_posteriors, statistics_versions):
            # x_o = compute_statistics_dict(observed_series["real"].values, s)
            samples = p.sample((100,))
            trajectories = np.array([run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW) for sample in samples])
            file_name = f"pngs/ppc_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_{','.join(to_short_names(s))}.png"
            ppc_trajectories(observed_series["real_gdp"].values[:T_hist+1], trajectories, file_name)
        
        ### for NN-based
        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            samples = p.sample((100,))
            trajectories = np.array([run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW) for sample in samples])
            file_name = f"pngs/ppc_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_nn{name}.png"
            ppc_trajectories(observed_series["real_gdp"].values[:T_hist+1], trajectories, file_name)

    if "ppc_stat" in argv:
        for p, s in zip(statistics_posteriors, statistics_versions):
            x_o = compute_statistics_dict(observed_series["real_gdp"].values[:T_hist], s)
            # samples = p.sample((100,))
            stats_out = np.array([compute_statistics_dict(run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW), s) for sample in samples])
            file_name = f"pngs/ppc_stat_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_npe_{','.join(to_short_names(s))}.png"
            ppc_plot(x_o, stats_out, file_name)
        
        for r, s in zip(statistics_nres, statistics_versions):
            x_o = compute_statistics_dict(observed_series["real_gdp"].values[:T_hist], s)
            stats_out = np.array([compute_statistics_dict(run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW), s) for sample in samples])
            file_name = f"pngs/ppc_stat_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_nre_{','.join(to_short_names(s))}.png"
            ppc_plot(x_o, stats_out, file_name)

    if "sbc" in argv:
        num_sbc_saples = 1000
        num_posterior_samples = 1000
        print(f"Running SBC with {num_sbc_saples} samples and {num_posterior_samples} posterior samples for each...")

        if not path.exists(f"data_npz/sbc_data_{num_sbc_saples}_{num_posterior_samples}.pkl"):
            sim_data, theta = gen_batch(
                FIRST_CALIBRATION_DATE, 
                NUM_CALIBRATION_DATES, 
                T_hist,
                priors, 
                parameters_to_calibrate, 
                num_sbc_saples, 
                n_runs=NUM_RUNS_PER_DRAW, 
                keys=keys
            )   
            with open(f"data_npz/sbc_data_{num_sbc_saples}_{num_posterior_samples}.pkl", "wb") as f:
                pickle.dump({"sim_data": sim_data, "theta": theta}, f)
        else:
            with open(f"data_npz/sbc_data_{num_sbc_saples}_{num_posterior_samples}.pkl", "rb") as f:
                data = pickle.load(f)
                sim_data = data["sim_data"]
                theta = data["theta"]

        # for p, s in zip(statistics_posteriors, statistics_versions):
        #     print(f"Running SBC for statistic set: {', '.join(to_short_names(s))}")
        #     reducted = reduction(sim_data).numpy()
        #     stats = np.array([compute_statistics_dict(sim_out, s) for sim_out in reducted])
        #     stats = torch.tensor(stats, dtype=torch.float32)
        #     ranks, dap_samples = run_sbc(
        #         theta, 
        #         stats, 
        #         p,
        #         num_posterior_samples=num_posterior_samples,
        #         use_batched_sampling=True,
        #         show_progress_bar=True,
        #         num_workers=4,
        #     )

        #     check_stats = check_sbc(
        #         ranks,
        #         prior_samples=theta,
        #         dap_samples=dap_samples,
        #         num_posterior_samples=num_posterior_samples,
        #     )

        #     print(
        #         f"SBC diagnostics [per dimension]:\nkolmogorov-smirnov p-values: {check_stats['ks_pvals'].numpy()}"
        #     )
        #     print(f"- c2st accuracies: {check_stats['c2st_ranks'].numpy()}")
        #     print(f"- c2st accuracies: {check_stats['c2st_dap'].numpy()}")

        #     fig, axes = sbc_rank_plot(
        #         ranks = ranks,
        #         num_posterior_samples=num_posterior_samples,
        #         plot_type="hist",
        #         num_bins=None,
        #     )
            
        #     plt.savefig(f"pngs/sbc_hist_nn_{name}_from_prior_{num_sbc_saples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
        #     plt.close()

        #     fig, axes = sbc_rank_plot(
        #         ranks = ranks,
        #         num_posterior_samples=num_posterior_samples,
        #         plot_type="cdf",
        #     )
            
        #     plt.savefig(f"pngs/sbc_cdf_nn_{name}_from_prior_{num_sbc_saples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
        #     plt.close()
        
        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            print(f"Running SBC for NN embedding: {name}")
            
            ranks, dap_samples = run_sbc(
                theta, 
                transform(sim_data).flatten(start_dim = 1),
                p,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=False,
                show_progress_bar=False,
            )

            check_stats = check_sbc(
                ranks,
                prior_samples=theta,
                dap_samples=dap_samples,
                num_posterior_samples=num_posterior_samples,
            )

            print(
                f"SBC diagnostics [per dimension]:\nkolmogorov-smirnov p-values: {check_stats['ks_pvals'].numpy()}"
            )
            print(f"- c2st accuracies: {check_stats['c2st_ranks'].numpy()}")
            print(f"- c2st accuracies: {check_stats['c2st_dap'].numpy()}")

            fig, axes = sbc_rank_plot(
                ranks = ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
            )
            
            plt.savefig(f"pngs/sbc_hist_nn_{name}_from_prior_{num_sbc_saples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
            plt.close()

            fig, axes = sbc_rank_plot(
                ranks = ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="cdf",
            )
            
            plt.savefig(f"pngs/sbc_cdf_nn_{name}_from_prior_{num_sbc_saples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
            plt.close()
        
    if "forecast" in argv:
        forecasts = {",".join(to_short_names(s)): forecast(p, final_params, final_initial) for p, s in zip(statistics_posteriors, statistics_versions)}
        forecasts.update({f"nn_{name}": forecast(p, final_params, final_initial) for p, name in zip(nn_posteriors, nn_versions)})
        forecasts.update({f"nre_{','.join(to_short_names(s))}": forecast(r, final_params, final_initial) for r, s in zip(statistics_nres, statistics_versions)})
        forecasts.update({f"nn_nre_{name}": forecast(r, final_params, final_initial) for r, name in zip(nn_nres, nn_versions)})
        
        abm_forecast = run_monte_carlo(final_params, final_initial, T_forecast, num_simulations=100)[1:]
        forecasts["ABM_base"] = abm_forecast

        if npe_x_base is not None and "base" in argv:
            forecast_statistic = torch.tensor(compute_statistics(observed_series["real_gdp"].values))
            npe_forecast_params = posterior.sample((1000, ), x = forecast_statistic).mean(dim= 0)
            npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
            npe_forecast = run_monte_carlo(npe_forecast_params, final_initial, T_forecast, num_simulations=100)[1:]

            forecasts["NPE_base"] = npe_forecast

        rolling_rsmfes = plot_forecasts(forecasts, realized_future, parameters_to_calibrate, bounds)

        print("Rolling RMSFEs:")
        for key, value in rolling_rsmfes.items():
            print(f"  {key}: {value}")