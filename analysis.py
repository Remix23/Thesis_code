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

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

T = 20
CALIBRATION_DATE = "2010-01-01"

### load from file
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


filename = ""

if not "load" in argv:
    numpy_files = [f for f in listdir(getcwd()) if f.endswith(".npz")]
    if len(numpy_files) > 0:
        print("Available .npz files:")
        for i, f in enumerate(numpy_files):
            print(f"{i}: {f}")
        idx = int(input("Enter the index of the file to load: "))
        filename = numpy_files[idx]
    else:
        print("No .npz files found. Please run the script with 'load' argument after generating data.")
        exit()

else:
    idx = argv.index("load") + 1
    if idx < len(argv):
        filename = argv[idx]
    else:
        print("Please provide the filename after 'load' argument.")
        exit()

if not filename.endswith(".npz"):
    print("Please provide a valid .npz file.")
    exit()

data = np.load(filename)
arrs = data.files
print(f"Loaded data from {filename} with attributes: ")
for i, arr in enumerate(arrs):
    print(f"{i}: {arr} ", end="\n")
print()

n_stats = None

def load_diff ():
    pass

prior_draws = torch.tensor(data["prior_draws"], dtype=torch.float32)
raw = torch.tensor(data["raw"], dtype=torch.float32)
to_calibrate = data["to_calibrate"]
parameters_to_calibrate = to_calibrate.tolist()
bounds = torch.tensor(data["bounds"], dtype=torch.float32)
if "statistics" in arrs:
    statistics = data["statistics"]
    n_stats = statistics.shape[1]

n_draws = prior_draws.shape[0]
parameters = jl.get_parameters()
initial_conditions = jl.get_initial_conditions()

### NPE
priors = BoxUniform(low=torch.tensor(bounds[:, 0], dtype=torch.float32), high=torch.tensor(bounds[:, 1], dtype=torch.float32))
inference = NPE(prior=priors, density_estimator='maf')
inference.append_simulations(prior_draws, raw)
inference.train()

posterior = inference.build_posterior()

# quartile data
real_gdp = pd.read_csv("data/austria_real_gdp_fred.csv", parse_dates=["observation_date"])
real_gdp["observation_date"] = pd.to_datetime(real_gdp["observation_date"])

### generate final prediction: (condition from 1995 to 2010)
conditional_period = real_gdp[real_gdp["observation_date"] < CALIBRATION_DATE]
conditional_values = conditional_period["CLVMNACSCAB1GQAT"].values
conditional_stats = compute_statistics(conditional_values)

if "ppc" in argv:
    ### posterio predictive checks
    samples = posterior.sample((100, ), x = conditional_stats)
    for i, sample in enumerate(samples):
        pass 

if "pairplot" in argv:
    _ = pairplot(
        posterior.sample((1000, ), x = conditional_stats).numpy(),
        labels=parameters_to_calibrate,
        points=np.array([parameters[param] for param in parameters_to_calibrate])[None, :],
        title = "Posterior distribution of calibrated parameters compared to calibrated parameter values",
    )

    plt.savefig(f"pngs/posterior_pairplot_p{len(parameters_to_calibrate)}_n{n_draws}_s{n_stats}.png")
    plt.clf()

if "sbc" in argv:
    ### simulation based calibration
    num_sbc_samples = 10
    thetas = priors.sample((num_sbc_samples,))
    params = [rep_parameters(parameters, theta) for theta in thetas]
    xs = [run_monte_carlo(param, initial_conditions, T, num_simulations=10) for param in params]
    stats = [compute_statistics(x) for x in xs]
    stats = torch.tensor(stats, dtype=torch.float32)
    sbc_results = run_sbc(
        posterior=posterior,
        thetas=thetas,
        xs = stats,
        num_posterior_samples=100
    )
    # check_sbc(sbc_results)

