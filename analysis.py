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
from sbi.inference import NPE, NPE_C, simulate_for_sbi
from sbi.analysis import pairplot
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import (
    CausalCNNEmbedding
)

from neural_network import RNN, CNN_GDP

from sbi.diagnostics import check_sbc, run_sbc

import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import pickle

torch.manual_seed(0)
np.random.seed(0)

NUM_SIM_PER_ROUND = 10
ROUNDS = 2

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

to_log_growth = lambda x: np.diff(np.log(x))
to_log = lambda x: np.log(x)
to_growth_rate = lambda x: np.diff(x) / x[:, :-1]

### x - (n_trajectories, T_hist) -> (n_trajectories, (T_hist - 1)*3) with channels log, growth rate, difference
# we want 
to_cnn3 = lambda x: np.concatenate([to_log(x)[:, 1:], to_growth_rate(x), np.diff(x)], axis=1)
# to_cnn3_channels = lambda x: np.concatenate([to_log(x)[:, 1:], to_growth_rate(x), np.diff(x)], axis=1)
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
        raise ValueError("Invalid file number selected.")

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

def train_npe_statistics_rounds (priors, stat_keys, full_params, initial_conditions, params_to_calibrate,observed, t_train, rounds = 3, n_sim_per_round = 100, ):
    
    inference = NPE_C(prior = priors)
    x_o = compute_statistics_dict(observed, stat_keys)
    x_o = torch.tensor(x_o, dtype=torch.float32)
    proposal = priors

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
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with statistics {', '.join(stat_keys)}")

    return posterior, proposals

def train_npe_nn_rounds (priors, nn, nn_transform, full_params, initial_conditions, params_to_calibrate,observed, t_train, rounds = 3, n_sim_per_round = 100):
    dense_estimator = posterior_nn(
        model="nsf",
        embedding_net=nn,
        z_score_x="structured"
    )
    x_o = torch.tensor(nn_transform(observed), dtype=torch.float32)
    inference = NPE_C(prior = priors, density_estimator=dense_estimator)
    proposal = priors
    proposals = []
    for i in range(rounds):
        proposals.append(proposal)
        if i > 0:
            theta = proposal.sample((n_sim_per_round, ), show_progress_bars=False)
        else:
            theta = proposal.sample((n_sim_per_round,))
        sim_data = np.array([run_monte_carlo(rep_parameters(full_params, params_to_calibrate, sample.numpy()), initial_conditions, t_train, num_simulations=1) for sample in theta])
        sim_data = sim_data.reshape((sim_data.shape[0], -1)) # reshape to (n_simulations, T_hist + 1) for CNN input
        x = torch.tensor(nn_transform(sim_data), dtype=torch.float32)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator=density_estimator)
        proposal = posterior.set_default_x(x_o)
        print(f"\nCompleted round {i + 1}/{rounds} of NPE training with NN embedding {nn.__class__.__name__}")
    return posterior, proposals

def train_npe_statistics (priors, prior_samples, sim_raw, stat_keys, rem_out = True):
    npe = NPE(prior=priors)
    npe_x = np.apply_along_axis(lambda sim_out: compute_statistics_dict(sim_out, stat_keys), 1, sim_raw)
    if rem_out:
        idx = remove_outliers(npe_x, m = 5, stat_names=stat_keys)
        npe_x = npe_x[idx, :]
        prior_samples = prior_samples[idx, :]
    npe_x = torch.tensor(npe_x, dtype=torch.float32)
    samples = torch.tensor(prior_samples, dtype=torch.float32)
    net = npe.append_simulations(samples, npe_x).train()
    posterior = npe.build_posterior()
    print()
    return posterior

def train_npe_nn (priors, prior_samples, sim_raw, nn):
    dens_estimator = posterior_nn(
        model="maf",
        embedding_net=nn,
        z_score_x="none"
    )
    npe = NPE(prior=priors, density_estimator=dens_estimator)
    npe_x = torch.tensor(sim_raw, dtype=torch.float32)
    samples = torch.tensor(prior_samples, dtype=torch.float32)
    net = npe.append_simulations(samples, npe_x).train()
    post = npe.build_posterior()
    return post

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

if __name__ == "__main__":

    priors_samples, npe_x_base, raw, parameters_to_calibrate, bounds = load_data()
    parameters_to_calibrate = parameters_to_calibrate.tolist()

    ### initial calibration
    initial_calibration = datetime(2010, 3, 31)

    n = 6 # number of years to simulate -> 5 years
    T_hist = 4 * n - 1 # quaters
    final_calibration = datetime(initial_calibration.year + T_hist // 4, 12, initial_calibration.day)

    T_forecast = 4 * 4 # 4 years forecast
    date_forecast = datetime(final_calibration.year + T_forecast // 4, final_calibration.month, final_calibration.day)
    ### real data
    csv_data = pd.read_csv("data_npz/italy_real_gdp.csv", parse_dates=["observation_date"])
    csv_data = csv_data[csv_data["observation_date"] >= initial_calibration]
    quarterly_dates, bit = [np.array(x) for x in jl.get_real([])]

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

    statistics_versions = []
    statistics_posteriors = []
    posteriors_hist = []

    for i, s in enumerate(statistics_versions):
        final_posterior, posteriors = train_npe_statistics_rounds(priors, s, hist_params, hist_initial, 
                                          parameters_to_calibrate, observed_series["real"].values, 
                                          t_train = T_hist, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND)
        posteriors_hist.append((s, posteriors))
        statistics_posteriors.append(final_posterior)

    # statistics_versions += [s_base + ["sequential"], s1 + ["sequential"]]

    ### networking embedding
    nns = []
    nn_posteriors = []
    input_transforms = []
    nn_versions = []

    if "nn" in argv:

        nn_raw = CausalCNNEmbedding(
            input_shape=(T_hist + 1, ), # length of the time series
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
            input_dim=T_hist + 1,
            flavour="elman",
            hidden_dim=32,
            num_layers=2,
            mlp_dims=12
        )

        s_nn = ["mean", "std", "yearly_corr", "ar1_coeff", "ar2_coeff", "min", "skewness", "kurtosis"]

        nn_cnn_mixture = CNN_GDP(stat_keys=s_nn, in_channels=3, conv_dims=[16, 32], summary_dims=16, out_dims=32, pool_kernel_size=1)

        nns += [nn_cnn_mixture]
        input_transforms +=  [to_cnn3]
        nn_versions += ["cnn_mixture"]

        assert len(nns) == len(input_transforms) == len(nn_versions)

    ### append NN-based posteriors
    print("Training NN-based NPEs...")
    observed_series_nn = observed_series["real"].values.reshape(1, -1) # reshape to (1, T_hist + 1) for CNN input
    for nn, transform, name in zip(nns, input_transforms, nn_versions):
        print(f"\nTraining NPE with NN embedding: {name}")
        post, posteriors = train_npe_nn_rounds(priors, nn, transform, hist_params, hist_initial, parameters_to_calibrate, observed_series_nn, T_hist, rounds=ROUNDS, n_sim_per_round=NUM_SIM_PER_ROUND)
        # post = train_npe_nn(priors, priors_samples, transform(raw), nn)
        nn_posteriors.append(post)
        nn_versions.append(f"nn_{name}")
        posteriors_hist.append((f"nn_{name}", posteriors))

    if "save" in argv:
        posteriors = {f"stat_{','.join(to_short_names(s))}": p for p, s in zip(statistics_posteriors, statistics_versions)}
        
        posteriors.update({f"nn_{name}": p for p, name in zip(nn_posteriors, nn_versions)})
        
        save_posteriors(posteriors, statistics_versions, f"trained_posteriors/posteriors_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}.pkl")
        save_posteriors(posteriors_hist, statistics_versions, f"trained_posteriors/posteriors_hist_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}.pkl")


    if "pp_forecast" in argv:
        ### pairplots for statistics-based NPE
        for p, s in zip(statistics_posteriors, statistics_versions):
            pplot_stat(p, final_params, filename = ",".join(to_short_names(s)) + "_forecast")
        
        ### for 
        for p, transform, name in zip(nn_posteriors, input_transforms, nn_versions):
            pplot_stat(p, final_params, filename = f"nn_{name}_forecast")

    
    if "pp_history" in argv:
        ### pairplots for statistics-based NPE
        for p, s in zip(statistics_posteriors, statistics_versions):
            pplot_stat(p, hist_params, filename = ",".join(to_short_names(s)) + "_hist")

        for p, transform, name in zip(nn_posteriors, input_transforms, nn_versions):
            pplot_stat(p, hist_params, filename = f"nn_{name}_hist")

    if "ppc" in argv:
        ### for statistics
        for p, s in zip(statistics_posteriors, statistics_versions):
            # x_o = compute_statistics_dict(observed_series["real"].values, s)
            samples = p.sample((100,))
            trajectories = np.array([run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=10) for sample in samples])
            file_name = f"pngs/ppc_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_{','.join(to_short_names(s))}.png"
            ppc_trajectories(observed_series["real"].values, trajectories, file_name)
        
        ### for NN-based
        for p, transform, name in zip(nn_posteriors, input_transforms, nn_versions):
            # x_o = transform(observed_series_nn)
            samples = p.sample((100,))
            trajectories = np.array([run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=10) for sample in samples])
            file_name = f"pngs/ppc_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_nn{name}.png"
            ppc_trajectories(observed_series["real"].values, trajectories, file_name)

    if "ppc_stat" in argv:
        for p, s in zip(statistics_posteriors, statistics_versions):
            x_o = compute_statistics_dict(observed_series["real"].values, s)
            # samples = p.sample((100,))
            stats_out = np.array([compute_statistics_dict(run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=10), s) for sample in samples])
            file_name = f"pngs/ppc_stat_r_{ROUNDS}_n{NUM_SIM_PER_ROUND}_p{len(parameters_to_calibrate)}_{','.join(to_short_names(s))}.png"
            ppc_plot(x_o, stats_out, file_name)

    if "sbc" in argv:
        num_sbc_saples = 100 * len(parameters_to_calibrate)
        prior_samples_sbc = priors.sample((num_sbc_saples,))
        for p, s in zip(statistics_posteriors, statistics_versions):
            print(f"Running SBC for statistic set: {', '.join(to_short_names(s))}")
            stats = np.array([compute_statistics_dict(run_monte_carlo(rep_parameters(hist_params, parameters_to_calibrate, sample.numpy()), hist_initial, T_hist, num_simulations=10), s) for sample in prior_samples_sbc])
            stats = torch.tensor(stats, dtype=torch.float32)
            sbc_results = run_sbc(
                prior_samples_sbc, 
                stats, 
                p,
                num_posterior_samples=1000,
            )
        
    if "forecast" in argv:
        forecasts = {",".join(to_short_names(s)): forecast(p, final_params, final_initial) for p, s in zip(statistics_posteriors, statistics_versions)}
        forecasts.update({f"nn_{name}": forecast(p, final_params, final_initial) for p, transform, name in zip(nn_posteriors, input_transforms, nn_versions)})
        
        abm_forecast = run_monte_carlo(final_params, final_initial, T_forecast, num_simulations=100)[1:]
        forecasts["ABM_base"] = abm_forecast

        print(forecasts.keys())
        print(forecasts.values())

        if npe_x_base is not None and "base" in argv:
            forecast_statistic = torch.tensor(compute_statistics(observed_series["real"].values))
            npe_forecast_params = posterior.sample((1000, ), x = forecast_statistic).mean(dim= 0)
            npe_forecast_params = rep_parameters(final_params, parameters_to_calibrate, npe_forecast_params)
            npe_forecast = run_monte_carlo(npe_forecast_params, final_initial, T_forecast, num_simulations=100)[1:]

            forecasts["NPE_base"] = npe_forecast

        rolling_rsmfes = plot_forecasts(forecasts, realized_future, parameters_to_calibrate, bounds)

        print("Rolling RMSFEs:")
        for key, value in rolling_rsmfes.items():
            print(f"  {key}: {value}")