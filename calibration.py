from juliacall import Main as jl

jl.seval("using JuliaModel: run_simulation, get_parameters, get_initial_conditions")

from sbi.utils import BoxUniform
from sbi.inference import NPE
import torch
import numpy as np

def run_sim(parameters, initial_conditions, T):
    data = jl.run_simulation(parameters, initial_conditions, T)
    return np.array(data)

def compute_statistics (sim_out):
    # compute summary statistics from the simulated data
    # for example, we can compute the mean and standard deviation of real GDP
    mean_gdp = np.mean(sim_out)
    std_gdp = np.std(sim_out)
    return np.array([mean_gdp, std_gdp])
### of statistics
n_stats = 2
n_draws = 10

parameters = jl.get_parameters()
initial_conditions = jl.get_initial_conditions()

### parameters:
parameters_to_calibrate = ["rho", "xi_pi", "xi_gamma", "alpha_G", "alpha_E", "alpha_I", "psi"]

priors_bounds = [
    (0.7, 0.99), # rho
    (1.0, 2.5),  # xi_pi
    (0.0, 1.0),    # xi_gamma
    (0.5, .99), (0.5, .99), (0.5, .99), # alpha_G, alpha_E, alpha_L
    (0.7, .99),    # psi
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

prior_draws = priors.sample((n_draws,))

npe_x = np.ndarray(shape = (n_draws, n_stats))

for i, draw in enumerate(prior_draws):
    print("Drawn parameters:", draw)
    sim_paramerters = parameters.copy()
    for param, value in zip(parameters_to_calibrate, draw.tolist()):
        print(f"  {param}: {value}")
        sim_paramerters[param] = value
    
    ### obtain simulated data
    x = run_sim(sim_paramerters, initial_conditions, 20)
    # compute summary statistics
    stats = compute_statistics(x)
    # print(f"Summary statistics for draw {i}:", stats)
    npe_x[i, :] = stats

### masked auto-regressive flow (MAF) density estimator
# elaborate on the choice of MAF and its advantages for this problem
inference = NPE(priors, density_estimator='maf')
inference.append_simulations(prior_draws, npe_x)
inference.train()

posterior = inference.build_posterior()