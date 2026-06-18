from os import path, listdir, getcwd, environ
from sys import argv
from datetime import datetime, date
from typing import Tuple
NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl
jl.seval("using JuliaModel: run_simulation, get_real, calibrate, run_monte_carlo, CalibrationData")
print(f"Using JuliaModel with {jl.seval('Threads.nthreads()')} threads.")

from sbi.utils import BoxUniform
from sbi.inference import NPE, NPE_C, simulate_for_sbi, NRE
from sbi.analysis import pairplot
from sbi.neural_nets import posterior_nn, classifier_nn
from sbi.neural_nets.embedding_nets import (
    CausalCNNEmbedding
)

from sbi.utils.diagnostics_utils import remove_nans_and_infs_in_x

from neural_network import RNN, CNN_GDP, SeqEmbedding, Hierarchical, SimpleHierarchical

from sbi.diagnostics import check_sbc, run_sbc
from sbi.analysis import sbc_rank_plot

import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import pickle

torch.manual_seed(0)
np.random.seed(0)

if "--seed" in argv:
    seed_index = argv.index("--seed") + 1
    if seed_index < len(argv):
        seed_value = int(argv[seed_index])
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        print(f"Set random seed to {seed_value} from command line argument.")
    else:
        print("Warning: --seed flag provided but no seed value found. Using default seed 0.")

FIRST_CALIBRATION_DATE = datetime(2010, 3, 31)
NUM_CALIBRATION_DATES = 5
NUM_SIM_PER_ROUND = 10

COUNTRY = "IT"

NUM_RUNS_PER_DRAW = 1

ROUNDS = 1

### num of simulations per round: NUM_SIM_PER_ROUND * NUM_CALIBRATION_DATES * NUM_RUNS_PER_DRAW for batched
### num of simulations per round for non-batched: NUM_SIM_PER_ROUND * NUM_RUNS_PER_DRAW

avaible_keys = [
    "real_gdp_quarterly",
    "nominal_gdp_quarterly",
    "real_gva_quarterly",
    "nominal_gva_quarterly",
    "gdp_deflator_quarterly",
    "real_household_consumption_quarterly", 
    "real_government_consumption_quarterly",
    "real_capitalformation_quarterly",
    "real_exports_quarterly",
    "real_imports_quarterly",
    "wages_quarterly",
]

loaded_theta_draws = None
loaded_sim_out = None

def rep_parameters(parameters, parameters_to_calibrate, draw):
    sim_paramerters = parameters.copy()
    for param, value in zip(parameters_to_calibrate, draw.tolist()):
        sim_paramerters[param] = value
    return sim_paramerters

def run_sim(parameters, initial_conditions, T, keys = ["real_gdp"]):
    data = jl.run_simulation(parameters, initial_conditions, T, keys)
    return np.array(data)

def run_monte_carlo(parameters, initial_conditions, T, num_simulations, calibration_date , keys, country):
    data = jl.run_monte_carlo(parameters, initial_conditions, T, num_simulations, keys, calibration_date, country)
    data = np.mean(data, axis=0) # average across simulations
    return data

def findar_p (times_series, p):
    target = times_series[p:]
    lagged = np.array([times_series[i:-(p - i)] for i in range(p)]).T
    ones = np.ones((lagged.shape[0], 1))
    lagged = np.hstack((ones, lagged))
    ols = np.linalg.lstsq(lagged, target, rcond=None)
    return ols[0]

def gen_sample (calibration_date, theta_draws, T, params_to_calibrate, n_runs, keys, country):
    params, initial_conditions = jl.calibrate(
        calibration_date.year, calibration_date.month, calibration_date.day, country
    )
    samples = torch.zeros((theta_draws.shape[0], n_runs, len(keys), T + 1))
    for i, draw in enumerate(theta_draws):
        sim_parameters = rep_parameters(params, params_to_calibrate, draw)
        sim_out = np.array(jl.run_monte_carlo(sim_parameters, initial_conditions, T, n_runs, keys, calibration_date, country))
        sim_out = torch.tensor(sim_out, dtype=torch.float32)
        samples[i, :, :, :] = sim_out

    return samples

def gen_batch (calibration_date, num_calibrations, T, priors, params_to_calibrate, n_samples, n_runs, keys, country, sample_theta=None, force_gen = False) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.zeros(
        n_samples, num_calibrations, n_runs, len(keys), T + 1
    )

    if "load" in argv and not force_gen:
        theta_draws = torch.tensor(loaded_theta_draws, dtype=torch.float32)
        batch = torch.tensor(loaded_sim_out, dtype=torch.float32)
        print(f"Loaded batch with shape {batch.shape} and theta_draws with shape {theta_draws.shape} from previous run.")
        idx = np.asarray([avaible_keys.index(key) for key in keys])
        batch = batch[:, :, :, idx, :]
        return theta_draws, batch

    theta_draws = torch.zeros((n_samples, len(params_to_calibrate)))

    if sample_theta is not None:
        theta_draws = sample_theta.repeat((n_samples, 1))
    else:
        if isinstance(priors, BoxUniform):
            theta_draws = priors.sample((n_samples, ))
        else:
            theta_draws = priors.sample((n_samples, ), show_progress_bars=False)

    for i in range(num_calibrations):
        cal_year = calibration_date.year + i
        cal_date = datetime(cal_year, calibration_date.month, calibration_date.day)
        print(f"Generating batch for calibration date: {cal_date.strftime('%Y-%m-%d')}")
        samples = gen_sample(cal_date, theta_draws, T, params_to_calibrate, n_runs, keys, country)
        batch[:, i, :, :, :] = samples

    return theta_draws, batch

def unrol_mc_runs (theta, sim_out):

    assert theta.shape[0] == sim_out.shape[0]
    ### shapes: 
    # theta: (n_samples, n_parameters)
    # sim_out: (n_samples, n_calibration_dates, n_runs, n_features, T + 1)

    ### out: (n_samples * n_runs, n_parameters), (n_samples * n_runs, n_calibration_dates, n_features, T + 1)
    n_samples, n_calibration_dates, n_runs, n_features, T_plus_1 = sim_out.shape
    theta_unrolled = theta.repeat_interleave(n_runs, dim=0) # (n_samples * n_runs, n_parameters)
    
    sim_out = sim_out.permute(0, 2, 1, 3, 4) # (n_samples, n_runs, n_calibration_dates, n_features, T + 1)
    sim_out_unrolled = sim_out.reshape(n_samples * n_runs, n_calibration_dates, n_features, T_plus_1) # (n_samples * n_runs, n_calibration_dates, n_features, T + 1)
    return theta_unrolled, sim_out_unrolled
    

def prepare_real (observed, keys, num_calibrations):
    ### we want observed to a dataframe with columns "date" and keys, and sorted by date, and with at least T_hist + 1 rows for each calibration date, and we want to return a generator of tensors of shape (len(keys), T_hist + 1) for each calibration date, where the tensor is the historical data for that calibration date.
    ### we output a x_o tensor for NN
    ### we need (calibration dates, n_feature, T_hist + 1)

    out = torch.zeros((num_calibrations, len(keys), T_hist + 1))

    for i in range(num_calibrations):
        cal_year = FIRST_CALIBRATION_DATE.year + i
        cal_date = datetime(cal_year, FIRST_CALIBRATION_DATE.month, FIRST_CALIBRATION_DATE.day)
        print(f"Preparing real data for calibration date: {cal_date.strftime('%Y-%m-%d')}")
        observed_cal = observed[observed["date"] >= cal_date]
        if len(observed_cal) < T_hist + 1:
            raise ValueError(f"Not enough historical data for calibration date {cal_date.strftime('%Y-%m-%d')}. Required: {T_hist + 1}, Available: {len(observed_cal)}")
        observed_cal = observed_cal.head(T_hist + 1)
        
    
        out[i, :, :] = torch.tensor(observed_cal[keys].values.T, dtype=torch.float32)

    return out

to_log_growth = lambda x: np.diff(np.log(x), axis=-1)

to_log_diff_torch = lambda x: torch.diff(torch.log(x), dim=-1)


to_log = lambda x: torch.log(x)
to_growth_rate = lambda x: torch.diff(x, dim=-1) / x[..., :-1]

### x - (n_trajectories, T_hist) -> (n_trajectories, (T_hist - 1)*3) with channels log, growth rate, difference
# we want 
to_cnn3 = lambda x: torch.concatenate([to_log(x)[:, 1:], to_growth_rate(x), torch.diff(x, dim=-1)], dim=-1)
# to_cnn3_channels = lambda x: np.concatenate([to_log(x)[:, 1:], to_growth_rate(x), np.diff(x)], axis=1)



### input transormations for NN-based NPEs
# x: (N_draws, N_calibration_dates, n_features, T_hist + 1)
to_one_series = lambda x: x[:, 0, 0, :] # take the first calibration date, first run, first feature

### 

STATISTICS = {
    "mean": lambda sim_out: np.mean(sim_out),
    "std": lambda sim_out: np.std(sim_out),
    "yearly_corr": lambda sim_out: np.corrcoef(sim_out[:-4], sim_out[4:])[0, 1],
    "auto_corr_1" : lambda sim_out: np.corrcoef(sim_out[:-1], sim_out[1:])[0, 1],
    "auto_corr_2" : lambda sim_out: np.corrcoef(sim_out[:-2], sim_out[2:])[0, 1],
    "auto_corr_3" : lambda sim_out: np.corrcoef(sim_out[:-3], sim_out[3:])[0, 1],
    "ar1_coeff": lambda sim_out: findar_p(sim_out, 1)[1],
    "ar2_coeff": lambda sim_out: findar_p(sim_out, 2)[1],
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
    ar2_coeff = findar_p(growth_rates, 2)[1]

    mini, maxi = np.min(growth_rates), np.max(growth_rates)
    
    skewness_gdp = np.mean((growth_rates - mean_gdprowth)**3) / std_gdprowth**3
    return np.array([mean_gdprowth, std_gdprowth, yearly_corr, ar1_coeff, ar2_coeff, mini, maxi, skewness_gdp])

def compute_statistics_dict (sim_out, keys = None):
    ### sim out: (n_features, T) -> (n_features, n_statistics)
    x = to_log_growth(sim_out)
    if keys is None:
        keys = STATISTICS.keys()
    stats = np.apply_along_axis(lambda sim_out: [STATISTICS[key](sim_out) for key in keys], axis=-1, arr=x)
    return stats

### load data
def load_custom ():
    parameters_to_calibrate = ["omega", "lambda_p", "pi_bar"]

    priors_bounds = [
        (0, 1),  # omega
        (0.001, 5),  # lambda_p
        (0, 1),  # pi_bar
    ]

    return parameters_to_calibrate, priors_bounds

def select_country():
    before_it_data_path = path.expanduser("~/Projects/julias/BeforeIT_Modded/data")
    counter = 1
    avaible_countries = []
    for item in listdir(before_it_data_path):
        if not path.isfile(path.join(before_it_data_path, item)) and len(item) == 2:
            print(f"{counter}. {item}")
            avaible_countries.append(item)
            counter += 1
    selection = int(input("Select the country for which to generate data (enter the number): ")) - 1

    country = avaible_countries[selection]
    print(f"Selected country: {country}")
    return country

def load_data():
    data_folder = path.join(getcwd(), "data_npz", "training")
    all_npe = listdir(data_folder) + listdir(getcwd())

    files = [f for f in all_npe if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError("No .npz files found in the data folder or current directory.")

    print("Available .npz files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    file_index = int(input("Enter the number corresponding to the file you want to load: ")) - 1

    if file_index < 0 or file_index >= len(files):
        return load_custom()

    file_name = files[file_index]
    file_path = path.join(data_folder, files[file_index]) if file_name in listdir(data_folder) else path.join(getcwd(), file_name)

    data = np.load(file_path)
    required_keys = ["theta_draw", "sim_out", "parameters_to_calibrate", "bounds"]
    print(f"Loaded data from {file_path} with keys: {list(data.files)}")
    if not all(key in data.files for key in required_keys):
        raise ValueError(f"Data file must contain the following keys: {required_keys}")

    print("Parameters: ")
    for param, value in zip(data["parameters_to_calibrate"], data["bounds"]):
        print(f"{param}: {value}")
    global loaded_theta_draws, loaded_sim_out, COUNTRY
    loaded_theta_draws = data["theta_draw"]
    loaded_sim_out = data["sim_out"]
    COUNTRY = file_name.split("_")[0] if "_" in file_name else COUNTRY

    if "n_runs" in data.files:
        global NUM_RUNS_PER_DRAW, NUM_CALIBRATION_DATES, FIRST_CALIBRATION_DATE, NUM_SIM_PER_ROUND
        NUM_RUNS_PER_DRAW = data["n_runs"].item()
        NUM_SIM_PER_ROUND = loaded_sim_out.shape[0]
        NUM_CALIBRATION_DATES = data["num_calibrations"].item()
        FIRST_CALIBRATION_DATE = datetime.strptime(data["starting_calibration_date"].item(), '%Y-%m-%d')

    print(f"[Settting global params]\nNUM_SIM_PER_ROUND to {NUM_SIM_PER_ROUND}\nNUM_RUNS_PER_DRAW to {NUM_RUNS_PER_DRAW}\nNUM_CALIBRATION_DATES to {NUM_CALIBRATION_DATES}\nFIRST_CALIBRATION_DATE to {FIRST_CALIBRATION_DATE.strftime('%Y-%m-%d')}\nCOUNTRY to {COUNTRY}")
    return data["parameters_to_calibrate"].tolist(), data["bounds"]


def save_posteriors (posteriors, verions, file_name):
    with open(file_name, "wb") as f:
        pickle.dump({
            "posteriors": posteriors, 
            "statistics_versions": verions,
            "country": COUNTRY,
            "num_calibration_dates": NUM_CALIBRATION_DATES,
            "num_sim_per_round": NUM_SIM_PER_ROUND,
            "rounds": ROUNDS,
            "first_calibration_date": FIRST_CALIBRATION_DATE.strftime('%Y-%m-%d'),
            "bounds": bounds,
            "parameters_to_calibrate": parameters_to_calibrate,
            }, f)
    print(f"Saved posteriors to {file_name}")

def mark_outliers(data: np.ndarray, m: float = 2.0, stat_names: list = None) -> np.ndarray:
    ### data is suppoer to 3dim: (n_samples, n_features, T), returns a boolean mask of the same shape as data, where True indicates that the corresponding value in data is not an outlier, and False indicates that it is an outlier.
    ### in: data: (n_samples, n_features, T)
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

def train_npe_nn_batched (priors, nn, nn_transform, params_to_calibrate, observed, t_train, features_kyes, country, rounds = 3, n_sim_per_round = 100, num_calibrations = NUM_CALIBRATION_DATES):
    dense_estimator = posterior_nn(
        model="nsf",
        embedding_net=nn,
        z_score_x="structured"
    )
    theta_transform, x_transform, x_o_transform = nn_transform
    x_o = x_o_transform(prepare_real(observed, features_kyes, num_calibrations).unsqueeze(0)).flatten() # (1, n_features, T_hist + 1) -> (1, embedding_dim)
    inference = NPE_C(prior = priors, density_estimator=dense_estimator, device="mps")
    proposal = priors
    for i in range(rounds):
        force_gen = True if i > 0 else False
        theta_draws, sim_out = gen_batch(FIRST_CALIBRATION_DATE, num_calibrations, 
                                       t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes, country, force_gen)
        
        # print(f"Generated batch with shape {batch.shape} for round {i + 1}/{rounds}")
        theta_draws, sim_out = unrol_mc_runs(theta_draws, sim_out)
        # print(f"Unrolled batch to shape {batch.shape} for round {i + 1}/{rounds}")

        batch = x_transform(sim_out) # (n_samples * n_runs, n_calibration
        theta_draws = theta_transform(theta_draws) # (n_samples * n_runs, n_parameters)
        
        # print(f"Transformed batch shape after NN transform: {batch.shape}")
        batch = batch.flatten(start_dim=1)

        print(f"Traning with:\n[theta_draws] shape: {theta_draws.shape}\n[batch] shape: {batch.shape}")

        density_estimator = inference.append_simulations(theta_draws, batch, proposal=proposal, data_device="cpu").train()
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with NN embedding {nn.__class__.__name__}")
    return posterior

def train_nre_nn_batched (priors, nn, nn_transform, params_to_calibrate, observed, t_train, features_kyes, country,  rounds = 3, n_sim_per_round = 100, num_calibrations = NUM_CALIBRATION_DATES):
    classifier = classifier_nn(
        model="mlp",
        embedding_net_x=nn,
        z_score_x="structured"
    )
    theta_transform, x_transform, x_o_transform = nn_transform
    x_o = x_o_transform(prepare_real(observed, features_kyes, num_calibrations).unsqueeze(0)).flatten() # (1, n_features, T_hist + 1) -> (1, embedding_dim)
    inference = NRE(prior = priors, classifier=classifier, device="mps")
    proposal = priors
    for i in range(rounds):
        force_gen = True if i > 0 else False
        theta_draws, sim_out = gen_batch(FIRST_CALIBRATION_DATE, num_calibrations, 
                                       t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes, country, force_gen)
        
        # print(f"Generated batch with shape {batch.shape} for round {i + 1}/{rounds}")
        theta_draws, sim_out = unrol_mc_runs(theta_draws, sim_out)
        # print(f"Unrolled batch to shape {batch.shape} for round {i + 1}/{rounds}")

        batch = x_transform(sim_out) # (n_samples * n_runs, n_calibration
        theta_draws = theta_transform(theta_draws) # (n_samples * n_runs, n_parameters)
        
        # print(f"Transformed batch shape after NN transform: {batch.shape}")
        batch = batch.flatten(start_dim=1)
        print(f"Traning with:\n[theta_draws] shape: {theta_draws.shape}\n[batch] shape: {batch.shape}")

        classifier = inference.append_simulations(theta_draws, batch, data_device="cpu").train()
        posterior = inference.build_posterior(density_estimator=classifier)
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NRE training with NN embedding {nn.__class__.__name__}")
    return posterior

def train_npe_statistics_rounds (priors, stat_keys, params_to_calibrate, observed, t_train, features_kyes, country, rounds = 3, n_sim_per_round = 100, num_calibrations = NUM_CALIBRATION_DATES):
    
    inference = NPE_C(prior = priors, device="mps")
    
    x_o = prepare_real(observed, features_kyes, num_calibrations) # (n_calibration_dates, n_features, T_hist + 1)
    x_o = x_o[-1, :, 1:] # take the last calibration date's data as x_o for the posterior, and add batch dimension -> (1, n_features, T_hist + 1)
    x_o = compute_statistics_dict(x_o.cpu().numpy(), stat_keys) # (n_features, n_statistics)
    x_o = torch.tensor(x_o, dtype=torch.float32)
    x_o = x_o.flatten(start_dim=0) # (n_features * n_statistics)

    npe_in = np.zeros((n_sim_per_round * NUM_RUNS_PER_DRAW * NUM_CALIBRATION_DATES, len(features_kyes), len(stat_keys)))

    proposal = priors
    for i in range(rounds):

        
        theta_draws, batch = gen_batch(FIRST_CALIBRATION_DATE, num_calibrations, 
                                       t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes, country)
        
        theta_draws, batch = unrol_mc_runs(theta_draws, batch)
        theta_draws = torch.repeat_interleave(theta_draws, NUM_CALIBRATION_DATES, dim=0)
        batch = batch.reshape(-1, batch.shape[2], batch.shape[3]) # (n_samples * n_runs, n_features, T_hist + 1)
        # idx = remove_outliers(batch, m = 8, stat_names=stat_keys)
        # theta = theta[idx, :]
        # x = x[idx, :]
        batch = batch.cpu().numpy()
        for j in range(batch.shape[0]):
            npe_in[j, :, :] = compute_statistics_dict(batch[j, :, :], stat_keys)
        x = torch.tensor(npe_in, dtype=torch.float32)
        x = x.flatten(start_dim=1) # (n_samples * n_runs, n_statistics)

        print(f"Traning NPE with:\n[theta_draws] shape: {theta_draws.shape}\n[batch] shape: {x.shape}")

        density_estimator = inference.append_simulations(theta_draws, x, proposal=proposal, data_device="cpu").train()
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with statistics {', '.join(stat_keys)}")

    return posterior

def train_nre_statistics_rounds (priors, stat_keys, params_to_calibrate, observed, t_train, features_kyes, country, rounds = 3, n_sim_per_round = 100, num_calibrations = NUM_CALIBRATION_DATES): 
    
    inference = NRE(prior = priors, device="mps")
    
    x_o = prepare_real(observed, features_kyes, num_calibrations) # (n_calibration_dates, n_features, T_hist + 1)
    x_o = x_o[-1, :, 1:] # take the last calibration date
    x_o = compute_statistics_dict(x_o.cpu().numpy(), stat_keys) # (n_features, n_statistics)
    x_o = torch.tensor(x_o, dtype=torch.float32)
    x_o = x_o.flatten(start_dim=0) # (n_features * n_statistics

    nre_in = np.zeros((n_sim_per_round * NUM_RUNS_PER_DRAW * NUM_CALIBRATION_DATES, len(features_kyes), len(stat_keys)))
   
    proposal = priors
    for i in range(rounds):
        theta_draws, batch = gen_batch(FIRST_CALIBRATION_DATE, num_calibrations, 
                                       t_train, proposal, params_to_calibrate, n_sim_per_round, NUM_RUNS_PER_DRAW, features_kyes, country)
        
        theta_draws, batch = unrol_mc_runs(theta_draws, batch)
        theta_draws = torch.repeat_interleave(theta_draws, NUM_CALIBRATION_DATES, dim=0)
        batch = batch.reshape(-1, batch.shape[2], batch.shape[3]) # (n_samples * n_runs, n_features, T_hist + 1)


        batch = batch.cpu().numpy()
        for j in range(batch.shape[0]):
            nre_in[j, :, :] = compute_statistics_dict(batch[j, :, :], stat_keys)
        x = torch.tensor(nre_in, dtype=torch.float32)
        x = x.flatten(start_dim=1) # (n_samples * n_runs, n_statistics)

        print(f"Traning NRE with:\n[theta_draws] shape: {theta_draws.shape}\n[batch] shape: {x.shape}")

        density_estimator = inference.append_simulations(theta, x, data_device = "cpu").train()
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        
        print(f"\nCompleted round {i + 1}/{rounds} of NRE training with statistics {', '.join(stat_keys)}")

    return posterior

def pplot_stat (posterior, params, filename):

    posterior_samples = posterior.sample((4000,)).cpu()

    post_mean = posterior_samples.mean(dim=0)
    post_dir = posterior_samples.std(dim=0)
    qs = torch.tensor([0.05, 0.5, 0.95])
    quantiles = torch.quantile(posterior_samples, qs, dim=0)

    print(f"Parameters: {parameters_to_calibrate}")
    print(f"Posterior mean: {"| ".join(f'{post_mean[i]:.4f}' for i, param in enumerate(parameters_to_calibrate))}")
    print(f"Posterior std: {"| ".join(f'{post_dir[i]:.4f}' for i, param in enumerate(parameters_to_calibrate))}")
    print(f"Posterior quantiles:")
    for i, q in enumerate(qs):
        print(f"  - {q:3.0%}: {"| ".join(f'{quantiles[i, j]:.4f}' for j, param in enumerate(parameters_to_calibrate))}")

    _ = pairplot(
        posterior_samples,
        points=np.array([params[param] for param in parameters_to_calibrate]),
        labels=list(parameters_to_calibrate),
        limits = torch.tensor(bounds, dtype=torch.float32),
        title = "Pairplot of the posterior samples with calibrated values"
    )
    ### store posterior samples in a csv file for later analysis
    with open(f"posterior_stats/stats_r{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_{filename}.csv", "w") as f:
        np.savetxt(f, posterior_samples, delimiter=",", header=",".join(parameters_to_calibrate), comments="")

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

def ppc_trajectories (observed_data, sim_out_trajectories, keys, country, filename):
    n_rows= 2
    ncols= len(keys) // n_rows + len(keys) % n_rows
    fig, axs = plt.subplots(2, len(keys) // 2 + len(keys) , figsize=(12, 4 * len(keys)))
    if len(keys) == 1:
        axs = [axs]

    for i, key in enumerate(keys):
        row = i // ncols
        col = i % ncols
        for sim_out in sim_out_trajectories:
            axs[row, col].plot(sim_out[i, :], color="blue", alpha=0.1)
        axs[row, col].plot(observed_data[key].values, color="red", label="Observed")
        axs[row, col].set_title(f"{key} trajectories vs observed")
        axs[row, col].legend()
    
    plt.title(f"Posterior Predictive Check - Trajectories for {country}")
    plt.savefig(filename)
    plt.close()

def load_sweep_data ():
    sweep_folder = path.join(getcwd(), "data_npz", "sweep")
    files = [f for f in listdir(sweep_folder) if f.endswith(".pkl")]
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    file_index = int(input("Enter the number corresponding to the sweep data file you want to load: ")) - 1
    file_path = path.join(sweep_folder, files[file_index])
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded sweep data from {file_path} with keys: {list(data.keys())}")
    return (
        data["sweep_data"],
        data["sweep_theta_values"],
        data["parameters_to_calibrate"],
        data["priors_bounds"],
        data["sim_per_point"],
        data["points_per_dim"],
        data["starting_calibration_date"],
        data["sweep_theta_base"],
    )

def gen_sweep_dataset (sim_per_point, points_per_dim,parameters_to_calibrate, priors_bounds, priors, country, keys):
    lowers, uppers = torch.tensor(priors_bounds, dtype=torch.float32).T
    theta_base = torch.tensor((lowers + uppers) / 2, dtype=torch.float32)

    assert len(theta_base) == len(parameters_to_calibrate), "Number of parameters to calibrate must match number of prior bounds."

    out = torch.zeros((
        len(theta_base), points_per_dim, sim_per_point, NUM_CALIBRATION_DATES, len(keys), T_hist + 1
    ))

    sweep_theta_values = torch.zeros((len(theta_base), points_per_dim))

    for i in range(len(theta_base)):
        low, high = priors_bounds[i]
        sweep_values = torch.linspace(low, high, points_per_dim + 2)[1:-1]
        sweep_theta_values[i, :] = sweep_values
        for j, val in enumerate(sweep_values):
            theta_sweep = theta_base.clone()
            theta_sweep[i] = val

            _, sweep_x = gen_batch(
                calibration_date=initial_calibration,
                num_calibrations=NUM_CALIBRATION_DATES,
                T=T_hist,
                priors = priors,
                params_to_calibrate=parameters_to_calibrate,
                n_samples=sim_per_point,
                n_runs=1,
                keys=keys,
                country=country,
                sample_theta = theta_sweep,
                force_gen=True
            )
            sweep_x = sweep_x.squeeze(2)
            ### we are left with (sim_per_point, num_calibrations, len(keys), T_hist + 1)
            out[i, j, :, :, :, :] = sweep_x
            print(f"Generated sweep data for parameter {parameters_to_calibrate[i]} at value {val:.4f} ({j + 1}/{points_per_dim})")

    return sweep_theta_values, theta_base, out

def sweep_posterior (posterior, bounds, sweep_data_x, sweep_data_theta, theta_base, parameters_to_calibrate, x_transform, sim_per_point, points_per_dim, posterior_draws, filename):
    fig, axes = plt.subplots(1, len(theta_base), figsize=(18, 5))

    for i, middle in enumerate(theta_base):
        param_name = parameters_to_calibrate[i]
        low, high = bounds[i]
        sweep_values = sweep_data_theta[i, :]
        points_out = np.zeros((len(sweep_values), 4)) ### (theta_k, mean, lower, upper) for a plot for a given parameter
        for j, val in enumerate(sweep_values):
            theta_sweep = theta_base.clone()
            theta_sweep[i] = val
            
            ### already removed MC dimensions along with generatio
            sweep_x = sweep_data_x[i, j, :, :, :, :]

            sweep_x = x_transform(sweep_x) 

            sweep_x = sweep_x.flatten(start_dim=1)

            sweep_x = sweep_x.to(posterior.device)
            post_samples = posterior.sample_batched((posterior_draws, ), x = sweep_x, show_progress_bars=False)

            ### clamp to prior

            # post_samples = post_samples.clamp(min=low, max=high)
            ### (post_draws, sim_per_point, n_parameters)

            post_samples = post_samples[:, :, i]

            ### Coverage:
            q1 = torch.quantile(post_samples, 0.05, dim=0)
            q3 = torch.quantile(post_samples, 0.95, dim=0)
            coverage = torch.mean(((val >= q1) & (val <= q3)).float()).item()
            # print(f"Sweep for parameter {param_name} at value {val:.4f}: 90% credible interval coverage: {coverage:.2%}")
            print(f"Sweep for parameter {param_name} at value {val:.4f}: 90% credible interval coverage: {coverage:.2%} (should be close to 90%)")

            ### compute sweeping, mean medians, for sim_per_point
            ### only for considered parameters dimension
            
            medians = torch.median(post_samples, dim=0).values
            median_mean = torch.mean(medians)
            points_out[j, 0] = val
            points_out[j, 1] = median_mean.item()

            ### uncertainty interval
            lower, upper = torch.quantile(medians, torch.tensor([0.05, 0.95], device=medians.device))
            points_out[j, 2] = lower.item()
            points_out[j, 3] = upper.item()

        axes[i].plot(points_out[:, 0], points_out[:, 1], marker="o")
        axes[i].fill_between(points_out[:, 0], points_out[:, 2], points_out[:, 3], color="blue", alpha=0.2, label="90% credible interval")
        ### add perfect line: y = x
        axes[i].plot([low, high], [low, high], color="green", linestyle="dashed", linewidth=2, label="Perfect forecast")
            
        ### x_limits
        axes[i].set_xlim(low, high)
        axes[i].set_ylim(low, high)

        axes[i].set_title(f"Sweep for parameter {param_name}")
        axes[i].set_xlabel(f"Value of {param_name} in the simulations")
        axes[i].set_ylabel(f"Mean of the medians of posterior samples for {param_name}")
        axes[i].legend()

    plt.suptitle(f"Sweep of posterior medians for each parameter with {posterior_draws} posterior samples and {sim_per_point} simulations per point")
    plt.savefig(f"pngs/{filename}")
    plt.close()


def coverage_analysis (theta_true, posterior_samples, levels : torch.Tensor):
    ### posterior_samples: (n_samples, n_parameters)
    n_samples, n_parameters = posterior_samples.shape
    coverage_results = {}

    lower = torch.quantile(posterior_samples, levels, dim=0) # (n_levels, n_parameters)
    upper = torch.quantile(posterior_samples, 1 - levels, dim=0) # (n_levels, n_parameters)

    coverage = 

    for level in levels
    return coverage_results


def forecast (posterior, final_params, final_initial, t_forecast, keys, calibration_date, country):

    posterior_samples = posterior.sample((1000, )).cpu()
    npe_forecast_params = posterior_samples.mean(dim= 0)
    npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
    npe_forecast = jl.run_monte_carlo(npe_forecast_params, final_initial, t_forecast, 100, keys, calibration_date, country)
    out = np.asarray(npe_forecast)

    return out.mean(axis=0)

def plot_forecasts (forecasts : dict[str, np.ndarray], realized_future, parameters_to_calibrate, bounds, keys, country):
    fig, axs = plt.subplots(2, len(keys) // 2 + len(keys) % 2, figsize=(18, 12))
    for i, key in zip(range(len(keys)), keys):
        ax = axs[np.unravel_index(i, axs.shape)]
        ax.plot(range(1, T_forecast + 1), realized_future[key], label="Real " + key, color="black")
        rolling_rsmfes = {}
        print(f"\nForecast comparison for {key}:")
        for version, forecast in forecasts.items():
            rmsfe = compute_rmsfes(forecast[i, 1:], realized_future[key].values)
            rolling_rsmfes[version] = rmsfe
            print(f"RMSFE of {version} forecast: {rmsfe[-1]:.4f}")
            ax.plot(range(1, T_forecast + 1), forecast[i, 1:], label=f"{version} ")
        
        ax.set_xticks(range(1, T_forecast + 1), realized_future["date"].dt.strftime("%Y-%m").values, rotation=45)
        ax.set_title(f"Real {key}")
        ax.set_xlabel("Time (quarters)")
        ax.set_ylabel(f"{key} value")
        ax.legend()
    plt.savefig(f"pngs/forecast_comparison_p{', '.join(parameters_to_calibrate)}_bounds{', '.join(str(b) for b in bounds)}.png")
    plt.show()
    return {}

def compute_rmsfes (forecast, realized):
    rolling_differences = [(forecast[:i] - realized[:i])**2 for i in range(1, len(forecast) + 1)]
    means = np.array(list(map(np.mean, rolling_differences)))
    return np.sqrt(means)

if __name__ == "__main__":

    if "load" in argv:
        parameters_to_calibrate, bounds = load_data()
    else:
        parameters_to_calibrate, bounds = load_custom()
    # parameters_to_calibrate = parameters_to_calibrate.tolist()


    ### statistics
    s1 = ["mean", "std", "yearly_corr", "ar1_coeff", "min", "max", "skewness", "quantile_50", "recession_count"]
    s1_prime = ["mean", "std", "yearly_corr", "ar1_coeff", "min", "skewness", "kurtosis", "recession_count"]
    s2 = ["mean", "std", "yearly_corr", "ar1_coeff", "ar2_coeff", "min", "skewness", "kurtosis"]
    s_base = ["mean", "std", "min", "max", "auto_corr_1", "auto_corr_2", "auto_corr_3","quantile_25", "quantile_50", "quantile_75"]

    s_base_seq = s_base + ["sequential"]
    s1_seq = s1 + ["sequential"]    

    ### NPE
    statistics_versions = [s_base]
    statistics_posteriors = []
    statistics_nres = []
    posteriors_hist = []

    ### networking embedding
    nns = []
    nn_posteriors = []
    nn_nres = []
    nn_transforms = []
    nn_versions = []
    nn_keys = []
    nn_cal_nums = []

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

        if path.exists(posteriors_path):
            with open(posteriors_path, "rb") as f:
                data = pickle.load(f)
            
            posteriors = data["posteriors"]
            bounds = data["bounds"]
            parameters_to_calibrate = data["parameters_to_calibrate"]

            COUNTRY = data["country"]
            ROUNDS = data["rounds"]
            NUM_SIM_PER_ROUND = data["num_sim_per_round"]
            NUM_CALIBRATION_DATES = data["num_calibration_dates"]
            FIRST_CALIBRATION_DATE = datetime.strptime(data["first_calibration_date"], "%Y-%m-%d")

            for key, posterior in posteriors.items():
                if key.startswith("stat_"):
                    statistics_posteriors.append(posterior)
                elif key.startswith("nn_"):
                    nn_posteriors.append(posterior)

            print(f"[NPE] Loaded {len(statistics_posteriors)} statistics-based posteriors\n[NPE] Loaded {len(nn_posteriors)} NN-based posteriors from {posterior_name}")
        
        if path.exists(ratio_path):
            with open(ratio_path, "rb") as f:
                data = pickle.load(f)
            nres = data["posteriors"]
            bounds = data["bounds"]
            parameters_to_calibrate = data["parameters_to_calibrate"]

            COUNTRY = data["country"]
            ROUNDS = data["rounds"]
            NUM_SIM_PER_ROUND = data["num_sim_per_round"]
            NUM_CALIBRATION_DATES = data["num_calibration_dates"]
            FIRST_CALIBRATION_DATE = datetime.strptime(data["first_calibration_date"], "%Y-%m-%d")

            for key, nre in nres.items():
                if key.startswith("stat_"):
                    statistics_nres.append(nre)
                elif key.startswith("nn_"):
                    nn_nres.append(nre)

            print(f"[NRE] Loaded {len(statistics_nres)} statistics-based NREs\n[NRE] Loaded {len(nn_nres)} NN-based NREs from {ratio_name}")
        
        print("[Setting global variables from loaded data]")
        print(f"COUNTRY: {COUNTRY}")
        print(f"ROUNDS: {ROUNDS}")
        print(f"NUM_SIM_PER_ROUND: {NUM_SIM_PER_ROUND}")
        print(f"NUM_CALIBRATION_DATES: {NUM_CALIBRATION_DATES}")
        print(f"FIRST_CALIBRATION_DATE: {FIRST_CALIBRATION_DATE.strftime('%Y-%m-%d')}")

     ### initial calibration
    print(f"Parameters to calibrate: {parameters_to_calibrate}")
    print(f"Bounds for parameters: {bounds}")
    priors = BoxUniform(low=torch.tensor(bounds, device="mps")[:, 0], high=torch.tensor(bounds, device="mps")[:, 1])
    # priors = torch.distributions.Uniform(low=torch.tensor(bounds, device="mps")[:, 0], high=torch.tensor(bounds, device="mps")[:, 1])

    initial_calibration = FIRST_CALIBRATION_DATE

    n = 3 # number of years to simulate -> 5 years
    T_hist = 4 * n - 1 # quaters
    final_calibration = datetime(initial_calibration.year + NUM_CALIBRATION_DATES, 12, initial_calibration.day)

    T_forecast = 4 * 3 # 4 years forecast
    date_forecast = datetime(final_calibration.year + T_forecast // 4, final_calibration.month, final_calibration.day)
    ### real data

    keys = [
        "real_gdp_quarterly",
        "gdp_deflator_quarterly",
        "real_household_consumption_quarterly",
        "real_government_consumption_quarterly",
        "real_capitalformation_quarterly",
    ]
    
    country = COUNTRY

    data, quarterly_dates = jl.get_real(keys, country)

    df = pd.DataFrame({"date": np.array(quarterly_dates).flatten(), **{key: np.array(data[key]).flatten() for key in keys}})

    df = df[df["date"] >= initial_calibration]

    observed_series = df[df["date"] <= final_calibration + pd.DateOffset(years=3)]

    realized_future = df[df["date"] > final_calibration]
    realized_future = realized_future.head(T_forecast)

    hist_params, hist_initial = jl.calibrate(initial_calibration.year, initial_calibration.month, initial_calibration.day, country)

    final_params, final_initial = jl.calibrate(final_calibration.year, final_calibration.month, final_calibration.day, country)

    hist_params["omega"] = 0.5
    final_params["omega"] = 0.5

    hist_params["lambda_p"] = 2
    final_params["lambda_p"] = 2

    hist_params["pi_bar"] = 1
    final_params["pi_bar"] = 1

    unrol_cal_dates = lambda x: x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

    to_seq_x = lambda x: to_log_diff_torch(x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
    to_seq_theta = lambda x: x.repeat_interleave(NUM_CALIBRATION_DATES, dim=0)
    ### take the last calibration date: (n_samples, calibration data, n feature, T)
    to_seq_x_o = lambda x: to_log_diff_torch(x[:, -1, :, :])

    to_hierarchical_x = lambda x: to_log_diff_torch(x)
    to_hierarchical_theta = lambda x: x

    if "nn" in argv:

        seq_multivariate = SeqEmbedding(
        T = T_hist,
            n_features = len(keys),
            hidden_size = 64,
            out_dim = 16
        )


        simple_hierarchical = SimpleHierarchical(
            calibration_dates=NUM_CALIBRATION_DATES,
            n_features=len(keys),
            T=T_hist,
            hidden_size=64,
            out_dim=16
        )

        hierarchical = Hierarchical(
            calibration_dates=NUM_CALIBRATION_DATES,
            n_features=len(keys),
            T=T_hist,
            hidden_sizes=[64, 32],
            out_dim=16
        )


        nns += [simple_hierarchical, hierarchical] # nn_raw, nn_diff, nn_3channels, nn_rnn, nn_cnn_mixture,
        x_transforms = [to_hierarchical_x, to_hierarchical_x] # to_seq_x, to_seq_x, to_seq_x, to_seq_x, to_seq_x,
        theta_transforms = [to_hierarchical_theta, to_hierarchical_theta] # to_seq_theta, to_seq_theta, to_seq_theta, to_seq_theta, to_seq_theta,
        x_o_transforms = [to_hierarchical_x, to_hierarchical_x] # to_seq_x_o, to_seq_x_o, to_seq_x_o, to_seq_x_o, to_seq_x_o,
        nn_transforms = list(zip(theta_transforms, x_transforms, x_o_transforms))
        # nn_transforms += [lambda x: reduction(x)[:, 1:], lambda x: to_growth_rate(reduction(x)), lambda x: x[..., 1:],  to_seq_multivariate]
        nn_versions += ["simple_hierarchical", "hierarchical"] # "cnn_log", "cnn_growth_rate", "cnn_level",
        nn_keys += [keys, keys] # [keys, keys, keys, keys, keys]
        nn_cal_nums += [NUM_CALIBRATION_DATES, NUM_CALIBRATION_DATES] # [NUM_CALIBRATION_DATES, NUM_CALIBRATION_DATES, NUM_CALIBRATION_DATES, NUM_CALIBRATION_DATES, NUM_CALIBRATION_DATES]

        assert len(nns) == len(nn_versions) == len(nn_transforms) == len(nn_keys), "Length of nns, nn_versions, nn_transforms and nn_keys must be the same"

    if not "load_posteriors" in argv:

        for i, s in enumerate(statistics_versions):
            final_posterior = train_npe_statistics_rounds(priors, s, parameters_to_calibrate, observed_series,                 
                                            t_train = T_hist, features_kyes=keys, country=country, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND, num_calibrations=NUM_CALIBRATION_DATES)
            statistics_posteriors.append(final_posterior)
        
        # for i, s in enumerate(statistics_versions):
        #     final_ratio, ratios = train_nre_statistics_rounds(priors, s, hist_params, hist_initial, 
        #                                     parameters_to_calibrate, observed_series.values[:T_hist + 1], 
        #                                     t_train = T_hist, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND)
            
        #     statistics_nres.append(final_ratio)

    ### append NN-based posteriors
    if not "load_posteriors" in argv:
        print("Training NN-based NPEs...")
        for nn, transform, name, key_list, cal_num in zip(nns, nn_transforms, nn_versions, nn_keys, nn_cal_nums):
            print(f"\nTraining NPE with NN embedding: {name}")
            post = train_npe_nn_batched(priors, nn, transform, parameters_to_calibrate, observed_series, T_hist, key_list, country, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND, num_calibrations=cal_num   )
            # post = train_npe_nn(priors, priors_samples, transform(raw), nn)
            nn_posteriors.append(post)

        # print("Training NN-based NREs...")
        # for nn, transform, name, key_list, cal_num in zip(nns, nn_transforms, nn_versions, nn_keys, nn_cal_nums):
        #     print(f"\nTraining NRE with NN embedding: {name}")
        #     post, posteriors = train_nre_nn_batched(priors, nn, transform, parameters_to_calibrate, observed_series, T_hist, key_list, country, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND, num_calibrations=cal_num)
        #     nn_nres.append(post)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "save" in argv:
        posteriors = {f"stat_{','.join(to_short_names(s))}": p for p, s in zip(statistics_posteriors, statistics_versions)}
        
        posteriors.update({f"nn_{name}": p for p, name in zip(nn_posteriors, nn_versions)})

        ratios = {f"stat_{','.join(to_short_names(s))}": r for r, s in zip(statistics_nres, statistics_versions)}
        ratios.update({f"nn_{name}": r for r, name in zip(nn_nres, nn_versions)})

        first_seed = np.random.get_state()[1][0]
        if len(posteriors) > 0:
            save_posteriors(posteriors, statistics_versions, f"trained_posteriors/posteriors_{COUNTRY}_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_{timestamp}_seed_{first_seed}.pkl")
        if len(ratios) > 0:
            save_posteriors(ratios, statistics_versions, f"trained_posteriors/ratios_{COUNTRY}_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_{timestamp}_seed_{first_seed}.pkl")

    if "pp" in argv:
        ### pairplots for statistics-based NPE
        for p, s in zip(statistics_posteriors, statistics_versions):
            pplot_stat(p, hist_params, filename = "npe_ " + ",".join(to_short_names(s)))

        for r, s in zip(statistics_nres, statistics_versions):
            pplot_stat(r, hist_params, filename = "nre_ " + ",".join(to_short_names(s)))

        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            pplot_stat(p, hist_params, filename = f"nn_npe_{name}")


        for r, transform, name in zip(nn_nres, nn_transforms, nn_versions):
            pplot_stat(r, hist_params, filename = f"nn_nre_{name}")

    if "ppc" in argv:
        ### for statistics
        for p, s in zip(statistics_posteriors, statistics_versions):
            # x_o = compute_statistics_dict(observed_series["real"].values, s)
            samples = p.sample((100,)).cpu()
            
            for year in range(NUM_CALIBRATION_DATES):
                cal_year= initial_calibration.year + year
                cal_date = datetime(cal_year, initial_calibration.month, initial_calibration.day)
                real_data_ten = observed_series[observed_series["date"] >= cal_date].iloc[:T_hist + 1, 1:].values.T
                
                trajectories = np.array([run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW, calibration_date=initial_calibration, keys = keys, country=country) for sample in samples])
                file_name = f"pngs/ppc_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_{','.join(to_short_names(s))}.png"
                ppc_trajectories(real_data_ten, trajectories, file_name)
        
        
        ### for NN-based
        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            samples = p.sample((100,)).cpu()
            
            for year in range(NUM_CALIBRATION_DATES):
                cal_year= initial_calibration.year + year
                cal_date = datetime(cal_year, initial_calibration.month, initial_calibration.day)
                real_data_ten = observed_series[observed_series["date"] >= cal_date].iloc[:T_hist + 1, 1:].values.T
                
                trajectories = np.array([run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW, calibration_date=initial_calibration, keys = keys, country=country) for sample in samples])
                file_name = f"pngs/ppc_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_nn_{name}.png"
                ppc_trajectories(real_data_ten, trajectories, file_name)

    if "ppc_stat" in argv:
        for p, s in zip(statistics_posteriors, statistics_versions):
            x_o = compute_statistics_dict(observed_series[keys].values[:T_hist], s)
            stats_out = np.array([compute_statistics_dict(run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW, calibration_date=initial_calibration, keys = keys, country=country), s) for sample in samples])
            file_name = f"pngs/ppc_stat_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_npe_{','.join(to_short_names(s))}.png"
            ppc_plot(x_o, stats_out, file_name)
        
        for r, s in zip(statistics_nres, statistics_versions):
            x_o = compute_statistics_dict(observed_series["real_gdp"].values[:T_hist], s)
            stats_out = np.array([compute_statistics_dict(run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=NUM_RUNS_PER_DRAW, calibration_date=initial_calibration, keys = keys, country=country), s) for sample in samples])
            file_name = f"pngs/ppc_stat_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_nre_{','.join(to_short_names(s))}.png"
            ppc_plot(x_o, stats_out, file_name)

    ### validation batches:
    num_prior_samples = 1000
    num_posterior_samples = 4000

    if not "load_validation" in argv:
        theta, sim_data = gen_batch(
            calibration_date=FIRST_CALIBRATION_DATE,
            num_calibrations=NUM_CALIBRATION_DATES,
            T=T_hist,
            priors=priors,
            params_to_calibrate=parameters_to_calibrate,
            n_samples=num_prior_samples,
            n_runs=1,
            keys=keys,
            country=country, 
            force_gen=True
        )
        with open(f"data_npz/validation/validation_{country}_p_{','.join(parameters_to_calibrate)}_b_{','.join([f'{low:.2f}_{high:.2f}' for low, high in bounds])}_prior_{num_prior_samples}_posterior_{num_posterior_samples}_c{NUM_CALIBRATION_DATES}_{timestamp}.pkl", "wb") as f:
            pickle.dump({"theta": theta, "sim_data": sim_data}, f)

    else:
        val_dir = "data_npz/validation"
        val_files = [f for f in listdir(val_dir) if f.endswith(".pkl")]
        for i, file in enumerate(val_files):
            print(f"{i + 1}. {file}")
        file_index = int(input("Enter the number corresponding to the validation dataset you want to load: ")) - 1
        
        if file_index < 0 or file_index >= len(val_files):
            print("Invalid index. Exiting.")
            exit()
        file_name = val_files[file_index]
        country_name = file_name.split("_")[1]

        assert country_name == country, f"Loaded validation dataset for country {country_name} does not match the current country {country}. Please select a valid dataset."

        val_path = path.join(val_dir, val_files[file_index])
        with open(val_path, "rb") as f:
            data = pickle.load(f)
        theta = data["theta"]
        sim_data = data["sim_data"]
        num_prior_samples = theta.shape[0]

    theta_draws, sim_out = unrol_mc_runs(theta, sim_data)
    theta_draws = theta_draws.to("mps")
    sim_out = sim_out.to("mps")
    print(f"Validation batch loaded with {theta_draws.shape[0]} parameter draws and simulation output of shape {sim_out.shape}")

    if "sweep" in argv or "load_sweep" in argv:

        sim_per_point = 25
        points_per_dim = 9
        posterior_draws = 4000

        if "load_sweep" in argv:
            sweep_data, sweep_theta_values, _parameters_to_calibrate, _bounds, sim_per_point, points_per_dim, _starting_calibration_date, theta_base = load_sweep_data()
        else:
            sweep_theta_values, theta_base, sweep_data = gen_sweep_dataset(sim_per_point, points_per_dim, parameters_to_calibrate=parameters_to_calibrate, priors_bounds=bounds, priors=priors, country=country, keys=keys)

            with open(f"data_npz/sweep/sweep_{country}_p_{','.join(parameters_to_calibrate)}_b_{','.join([f'{low:.2f}_{high:.2f}' for low, high in bounds])}_sim_{sim_per_point}_prior_{num_prior_samples}_posterior_{num_posterior_samples}_c{NUM_CALIBRATION_DATES}_{timestamp}.pkl", "wb") as f:
                pickle.dump({
                    "sweep_data": sweep_data,
                    "sweep_theta_values": sweep_theta_values,
                    "parameters_to_calibrate": parameters_to_calibrate,
                    "priors_bounds": bounds,
                    "sim_per_point": sim_per_point,
                    "points_per_dim": points_per_dim,
                    "starting_calibration_date": FIRST_CALIBRATION_DATE,
                    "sweep_theta_base" : theta_base
                }, f)

        
        for p, s in zip(statistics_posteriors, statistics_versions):
            print(f"Running sweep for statistics embedding: {','.join(to_short_names(s))}")
            sweep_posterior(
                posterior=p, 
                bounds=bounds, 
                sweep_data_x=sweep_data,
                sweep_data_theta=sweep_theta_values,
                theta_base=theta_base,
                parameters_to_calibrate=parameters_to_calibrate, 
                ### x is four dim: (sim_per_point, num_calibrations, len(keys), T_hist + 1)
                x_transform=lambda x: torch.stack([torch.tensor(compute_statistics_dict(batch, s), dtype=torch.float32) for batch in unrol_cal_dates(x).cpu().numpy()], dim=0).to(p.device),
                sim_per_point=sim_per_point,
                points_per_dim=points_per_dim,
                posterior_draws=posterior_draws,
                filename=f"sweep_posterior_medians_stat_{','.join(to_short_names(s))}_from_posterior_{posterior_draws}_{timestamp}.png"
            )

        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            print(f"Running sweep for NN embedding: {name}")
            sweep_posterior(
                posterior=p, 
                bounds=bounds, 
                sweep_data_x=sweep_data,
                sweep_data_theta=sweep_theta_values,
                theta_base=theta_base,
                parameters_to_calibrate=parameters_to_calibrate, 
                x_transform=transform[1],
                sim_per_point=sim_per_point,
                points_per_dim=points_per_dim,
                posterior_draws=posterior_draws,
                filename=f"sweep_posterior_medians_nn_{name}_from_posterior_{posterior_draws}_{timestamp}.png"
            )

    ### validation

    if "coverage" in argv:
        
        for p, s in zip(statistics_posteriors, statistics_versions):
            print(f"Running coverage analysis for statistics embedding: {','.join(to_short_names(s))}")
            coverage_analysis(
                posterior=p,
                bounds=bounds,
                sim_out=sim_out,
                theta_draws=theta_draws,
                parameters_to_calibrate=parameters_to_calibrate,
                x_transform=lambda x: torch.stack([torch.tensor(compute_statistics_dict(batch, s), dtype=torch.float32) for batch in unrol_cal_dates(x).cpu().numpy()], dim=0).to(p.device),
                filename=f"coverage_stat_{','.join(to_short_names(s))}_from_posterior_{num_posterior_samples}_{timestamp}.png"
            )

    if "sbc" in argv:

        for p, transform, name in zip(nn_posteriors, nn_transforms, nn_versions):
            print(f"Running SBC for NN embedding: {name}")
            theta_transform, x_transform, x_o_transform = transform

            xs = x_transform(sim_out).flatten(start_dim=1)
            xs = xs[::sim_out.shape[1], :]
            theta, xs = remove_nans_and_infs_in_x(theta_draws, xs)
            
            ranks, dap_samples = run_sbc(
                thetas=theta,
                xs=xs,
                posterior=p,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=False,
                show_progress_bar=True,
            )

            check_stats = check_sbc(
                ranks = ranks,
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
                parameter_labels=parameters_to_calibrate
            )
            
            plt.savefig(f"pngs/sbc_hist_nn_{name}_from_prior_{num_prior_samples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
            plt.close()

            fig, axes = sbc_rank_plot(
                ranks = ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="cdf",
                parameter_labels=parameters_to_calibrate
            )
            
            plt.savefig(f"pngs/sbc_cdf_nn_{name}_from_prior_{num_prior_samples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
            plt.close()
        
        for p, s in zip(statistics_posteriors, statistics_versions):
            name = ",".join(to_short_names(s))

            ### sim out: (sim_per_point, num_calibrations, len(keys), T_hist + 1)
            print(f"Running SBC for statistics embedding: {s}")

            # to_keep = torch.arange(0, sim_out.shape[0], step=sim_out.shape[1], dtype=torch.int32)
            xs = unrol_cal_dates(sim_out)
            xs = xs.cpu().numpy()
            
            sbc_in = torch.zeros(sim_out.shape[0], sim_out.shape[2], len(s)) ### (num_prior_samples * num_calibrations, len(keys), len(statistics))
            for i in range(sbc_in.shape[0]):
                stat_i = compute_statistics_dict(xs[i * sim_out.shape[1], :, :], s)
                sbc_in[i, :, :] = torch.tensor(stat_i, dtype=torch.float32)
            
            sbc_in = sbc_in.flatten(start_dim=1)
            
            theta, sbc_in = remove_nans_and_infs_in_x(theta, sbc_in)
            sbc_in = sbc_in.to(posterior.device)

            ranks, dap_samples = run_sbc(
                thetas=theta,
                xs=sbc_in,
                posterior=p,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=False,
                show_progress_bar=True,
            )

            check_stats = check_sbc(
                ranks = ranks,
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
                parameter_labels=parameters_to_calibrate
            )
            
            plt.savefig(f"pngs/sbc_hist_nn_{name}_from_prior_{num_prior_samples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
            plt.close()

            fig, axes = sbc_rank_plot(
                ranks = ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="cdf",
                parameter_labels=parameters_to_calibrate
            )
            
            plt.savefig(f"pngs/sbc_cdf_nn_{name}_from_prior_{num_prior_samples}_from_posterior_{num_posterior_samples}_{timestamp}.png")
            plt.close()


    if "forecast" in argv:
        forecasts = {",".join(to_short_names(s)): forecast(p, final_params, final_initial, T_forecast, keys, final_calibration, country) for p, s in zip(statistics_posteriors, statistics_versions)}
        forecasts.update({f"nn_{name}": forecast(p, final_params, final_initial, T_forecast, keys, final_calibration, country) for p, name in zip(nn_posteriors, nn_versions)})
        forecasts.update({f"nre_{','.join(to_short_names(s))}": forecast(r, final_params, final_initial, T_forecast, keys, final_calibration, country) for r, s in zip(statistics_nres, statistics_versions)})
        forecasts.update({f"nn_nre_{name}": forecast(r, final_params, final_initial, T_forecast, keys, final_calibration, country) for r, name in zip(nn_nres, nn_versions)})
        
        abm_forecast = run_monte_carlo(final_params, final_initial, T_forecast, num_simulations=100, calibration_date=final_calibration, keys=keys, country=country)
        forecasts["ABM_base"] = abm_forecast

        rolling_rsmfes = plot_forecasts(forecasts, realized_future, parameters_to_calibrate, bounds, keys, country)

        print("Rolling RMSFEs:")
        for key, value in rolling_rsmfes.items():
            print(f"  {key}: {value}")