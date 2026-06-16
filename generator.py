from datetime import date, datetime
from os import environ, getcwd, listdir, path
from sys import argv
from typing import Tuple
import pickle

NUM_THREADS = 10

environ["JULIA_NUM_THREADS"] = str(NUM_THREADS)
environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl

jl.seval("using JuliaModel: run_simulation, get_real, calibrate, run_monte_carlo, CalibrationData")
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
num_calibrations = 5

n = 3  # number of years to simulate -> 3 years
T_hist = 4 * n - 1  # quaters
final_calibration = datetime(
    initial_calibration.year + num_calibrations + T_hist // 4, 12, initial_calibration.day
)

T_forecast = 4 * 3  # 3 years forecast
date_forecast = datetime(
    final_calibration.year + T_forecast // 4,
    final_calibration.month,
    final_calibration.day,
)

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


hist_params, hist_initial = jl.calibrate(
    initial_calibration.year, initial_calibration.month, initial_calibration.day, country
)

final_params, final_initial = jl.calibrate(
    final_calibration.year, final_calibration.month, final_calibration.day, country
)
### gen historical date for NPE

# parameters_to_calibrate = ["zeta_b", "omega", "lambda_p"]
parameters_to_calibrate = ["omega", "lambda_p", "pi_bar"]

priors_bounds = [
    (0, 1),  # omega
    (0.001, 5),  # lambda_p
    (.5, 1.2),  # pi_bar
]

assert len(parameters_to_calibrate) == len(priors_bounds), "Number of parameters to calibrate must match number of prior bounds."

for param, bounds in zip(parameters_to_calibrate, priors_bounds):
    calibrated = hist_params.get(param, None)
    if calibrated is None: 
        print(f"Parameter {param} with bounds {bounds} not found in historical calibration.")
        continue
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

def run_prior_check(samples, real, cal_date, keys):
    ### samples: (n_samples, n_runs, len(keys), T + 1)
    # real: (len(keys), T + 1)
    # average accross runs
    avg_samples = np.mean(samples, axis=1)  # (n_samples, len
    n_rows = int(np.ceil(len(keys) / 4))
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 16))
    for i in range(avg_samples.shape[1]): ### for all keys
        idx = np.unravel_index(i, (n_rows, n_cols))
        plt.sca(axs[idx])
        plt.plot(real[i, :], label="Real", color="black", linewidth=2)
        for j in range(avg_samples.shape[0]): ### for all samples
            plt.plot(avg_samples[j, i, :], alpha=0.5, color="blue")
        plt.title(f"Prior check for key: {keys[i]}")
        plt.xlabel(f"Time, from {cal_date.strftime('%Y-%m-%d')}")
        plt.ylabel(keys[i])
    file_end = f"p_{",".join(parameters_to_calibrate)}_bounds{','.join([str(b) for b in priors_bounds.numpy()])}"
    plt.savefig(f"pngs/prior_checks/{country}_prior_check_{cal_date.strftime('%Y-%m-%d')}_{file_end}.png")
    plt.legend(loc="upper left")
    plt.close()
    
    
def gen_sample (calibration_date, theta_draws, T, params_to_calibrate, n_runs, keys, country) -> torch.Tensor:
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

def gen_batch (calibration_date, num_calibrations, T, priors, params_to_calibrate, n_samples, n_runs, keys, country, sample_theta=None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.zeros(
        n_samples, num_calibrations, n_runs, len(keys), T + 1
    )
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

def gen_sweep_dataset (parameters_to_calibrate, priors_bounds, priors, country):
    lowers, uppers = torch.tensor(priors_bounds, dtype=torch.float32).T
    theta_base = torch.tensor((lowers + uppers) / 2, dtype=torch.float32)

    assert len(theta_base) == len(parameters_to_calibrate), "Number of parameters to calibrate must match number of prior bounds."

    sim_per_point = 20
    points_per_dim = 9

    out = torch.zeros((
        len(theta_base), points_per_dim, sim_per_point, num_calibrations, len(avaible_keys), T_hist + 1
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
                num_calibrations=num_calibrations,
                T=T_hist,
                priors = priors,
                params_to_calibrate=parameters_to_calibrate,
                n_samples=sim_per_point,
                n_runs=1,
                keys=avaible_keys,
                country=country,
                sample_theta = theta_sweep,
            )
            sweep_x = sweep_x.squeeze(2)
            ### we are left with (sim_per_point, num_calibrations, len(keys), T_hist + 1)
            out[i, j, :, :, :, :] = sweep_x
    ### to file
    file_name = f"{country}_sweep_{','.join(parameters_to_calibrate)}_bounds{','.join([str(b) for b in priors_bounds])}.pkl"
    with open(path.join("data_npz", "sweep", file_name), "wb") as f:
        pickle.dump(
            {
                "sweep_data": out.numpy(),
                "sweep_theta_values": sweep_theta_values.numpy(),
                "parameters_to_calibrate": parameters_to_calibrate,
                "priors_bounds": priors_bounds.numpy(),
                "num_calibrations": num_calibrations,
                "sim_per_point": sim_per_point,
                "points_per_dim": points_per_dim,
                "starting_calibration_date": initial_calibration.strftime('%Y-%m-%d')
            },
            f
        )
    return sweep_theta_values, out

n_runs = 5
keys = avaible_keys
samples = np.zeros((n_histories, num_calibrations, n_runs, len(keys), T_hist + 1))
prior_draws = np.zeros((num_calibrations, n_histories, len(parameters_to_calibrate)))

### real data

data, quarterly_dates = jl.get_real(keys, country)

n = len(quarterly_dates)
df = pd.DataFrame({"date": np.array(quarterly_dates).flatten(), **{key: np.array(data[key]).flatten() for key in keys}})

df = df[df["date"] >= initial_calibration]

prior_draws, samples = gen_batch(initial_calibration, num_calibrations, T_hist, priors, parameters_to_calibrate, n_histories, n_runs, keys, country)

if "prior_check" in argv:
    for i in range(num_calibrations):
        cal_year = initial_calibration.year + i
        cal_date = datetime(cal_year, initial_calibration.month, initial_calibration.day)
        real_data_ten = df[df["date"] >= cal_date].iloc[:T_hist + 1, 1:].values.T
        run_prior_check(samples[:, i, :, :, :].numpy(), real_data_ten, cal_date, keys)

print(df.head(n = 20))

nans = np.isnan(samples.numpy())
is_row_nan = np.any(nans, axis=(2, 3, 4))  # check if any value in the row is NaN
print(f"Number of NaNs in samples: {is_row_nan.sum()} out of {nans.size} total values")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

### save
np.savez(
    f"data_npz/training/{country}_prior_samples_n{n_histories}_{','.join(parameters_to_calibrate)}_bounds{','.join([str(b) for b in priors_bounds.numpy()])}_{timestamp}.npz",
    sim_out=samples,
    theta_draw = prior_draws,
    bounds = priors_bounds.numpy(),
    parameters_to_calibrate = parameters_to_calibrate,
    num_calibrations = num_calibrations,
    n_runs = n_runs,
    starting_calibration_date = initial_calibration.strftime('%Y-%m-%d'),
)

