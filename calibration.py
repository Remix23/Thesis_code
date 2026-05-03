from os import environ

NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl

jl.seval("using JuliaModel: run_simulation, get_parameters, get_initial_conditions, run_for_different_parameters, run_monte_carlo")
print(f"Using JuliaModel with {jl.seval('Threads.nthreads()')} threads.")

from sbi.utils import BoxUniform
from sbi.inference import NPE
import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from os import path
from sys import argv

from time import time

### for now, only real GDP
CALIBRATION_DATE = "2010-01-01"
T = 20

def rep_parameters(parameters, draw):
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

def compute_statistics (sim_out):
    # compute summary statistics from the simulated data
    # for example, we can compute the mean and standard deviation of real GDP

    mean_gdp = np.mean(sim_out)
    std_gdp = np.std(sim_out)

    ### yearly correlation y_t and y_t-4
    yearly_corr = np.corrcoef(sim_out[:-4], sim_out[4:])[0, 1]
    ### AR(1) coefficient of real GDP
    lagged = sim_out[:-1]
    target = sim_out[1:]
    ols = np.polyfit(lagged, target, 1)
    ar1_coeff = ols[0]
    ### skewness of real GDP
    # skewness_gdp = np.mean((sim_out - mean_gdp)**3) / std_gdp**3
    return np.array([mean_gdp, std_gdp, yearly_corr, ar1_coeff])
### of statistics
n_stats = 4
n_draws = 200
parameters = jl.get_parameters()

not_floats = [(parameters[param].shape, param) for param in parameters if not isinstance(parameters[param], float)]
print(not_floats)
print(np.sum(parameters["I_s"]))
exit()
initial_conditions = jl.get_initial_conditions()

### parameters:
beta0 = parameters["beta_E"]

# parameters_to_calibrate = ["rho", "alpha_G", "alpha_E", "beta_E", "xi_gamma"]
parameters_to_calibrate = ["psi", "alpha_G", "xi_gamma"]

priors_bounds = [
    (0.7, 0.99), # psi
    (0.7, .99), # alpha_G,
    # (0.5, .99), # alpha_E
    (.7, 1.5),    # xi_gamma
]

for param, bounds in zip(parameters_to_calibrate, priors_bounds):
    calibrated = parameters[param]
    if calibrated <= bounds[0] or calibrated >= bounds[1]:
        print(f"Warning: Parameter {param} with value {calibrated} is outside the bounds {bounds}") 
    else:
        print(f"Parameter {param} with value {calibrated} is within the bounds {bounds}")

### construct priors
priors_bounds = torch.tensor(priors_bounds, dtype=torch.float32)
priors = BoxUniform(low=priors_bounds[:, 0], high=priors_bounds[:, 1])

npe_x = np.ndarray(shape = (n_draws, n_stats))

### simulate data for NPE construction

if path.exists("npe_data.npz") and len(argv) > 1 and argv[1] == "load":
    print("Loading existing NPE data from file...")
    data = np.load("npe_data.npz")
    prior_draws = data["prior_draws"]
    npe_x = data["npe_x"]
else:
    print("Generating new NPE data...")
    prior_draws = priors.sample((n_draws,))
    for i, draw in enumerate(prior_draws):
        sim_paramerters = rep_parameters(parameters, draw)
        
        ### obtain simulated data
        x = run_monte_carlo(sim_paramerters, initial_conditions, T, num_simulations=10)
        # compute summary statistics
        stats = compute_statistics(x)
        # print(f"Summary statistics for draw {i}:", stats)
        npe_x[i, :] = stats

        print(f"Completed {i+1}/{n_draws} simulations for NPE data generation", end="\r")
    
    ## save
    np.savez("npe_data.npz", prior_draws=prior_draws.numpy(), npe_x=npe_x)

### masked auto-regressive flow (MAF) density estimator
# elaborate on the choice of MAF and its advantages for this problem

# to tensor
prior_draws = torch.tensor(prior_draws, dtype=torch.float32)
npe_x = torch.tensor(npe_x, dtype=torch.float32)

inference = NPE(priors, density_estimator='maf')
inference.append_simulations(prior_draws, npe_x)
inference.train()

posterior = inference.build_posterior()

print("Posterior sampling...")
post_mean = posterior.sample((1000, ), x=npe_x.mean(dim=0)).mean(dim=0)
print("Posterior mean:", post_mean)

### simulation based calibration


### validation
# quartile data
real_gdp = pd.read_csv("data/austria_real_gdp_fred.csv", parse_dates=["observation_date"])
real_gdp["observation_date"] = pd.to_datetime(real_gdp["observation_date"])
real_gdp = real_gdp[real_gdp["observation_date"] >= CALIBRATION_DATE]
real_gdp = real_gdp["CLVMNACSCAB1GQAT"].values

final_params = rep_parameters(parameters, post_mean)

prediction = run_monte_carlo(final_params, initial_conditions, T, num_simulations=50)

# Save prediction to CSV
calibration_date = pd.to_datetime(CALIBRATION_DATE)
date_range = pd.date_range(start=calibration_date, periods=len(prediction), freq='QS')
prediction_df = pd.DataFrame({
    'date': date_range,
    'prediction': prediction
})
prediction_df.to_csv('prediction.csv', index=False)

plt.plot(prediction, label="Prediction")
plt.plot(real_gdp[:T], label="Real GDP")
plt.legend()
plt.title("Model Prediction vs Real GDP")
plt.show()