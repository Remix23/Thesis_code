import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sys import argv
from os import path, listdir
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR


def forecast_ar1(coef, steps, initial_value):
    forecast = [initial_value]
    for _ in range(steps):
        next_value = coef[0] + coef[1] * forecast[-1]
        forecast.append(next_value)
    return np.array(forecast[1:])  # Exclude the initial value
    
### load all posterios for seeds
def findar_p (times_series, p):
    target = times_series[p:]
    lagged = np.column_stack([
        times_series[p - lag: len(times_series) - lag]
        for lag in range(1, p + 1)
    ])
    ones = np.ones((lagged.shape[0], 1))
    lagged = np.hstack((ones, lagged))
    ols = np.linalg.lstsq(lagged, target, rcond=None)
    return ols[0]

### load real data:
data_dir = path.join(path.dirname(__file__), "data_npz", "real_data")
files = [f for f in listdir(data_dir) if f.endswith(".csv")]
print(f"Found {len(files)} CSV files in {data_dir}:")
for i, f in enumerate(files):
    print(f" - {i + 1}: {f}")
selection = int(input("Enter the number of the CSV file to load (or press Enter to skip): ")) - 1
if selection >= 0:
    file_path = path.join(data_dir, files[selection])

    if not path.isfile(file_path):
        print(f"File {selection} not found in {data_dir}.")
    else:
        real_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Loaded data from {selection}: shape {real_data.shape}")

keys = [
        "real_gdp_quarterly",
        "gdp_deflator_quarterly",
        "real_household_consumption_quarterly",
        "real_government_consumption_quarterly",
        "real_capitalformation_quarterly",
    ]

real_data = real_data[keys].dropna()
print(real_data.head())

forecast_from = pd.to_datetime("2016-12-31")
start_from = len(real_data[real_data.index < forecast_from])
print(f"Starting from index {start_from} for forecasting (date: {real_data.index[start_from]})")
time_series = real_data[real_data.index <= forecast_from].values.T
print(f"Shape of time series: {time_series.shape}")
realized_series = real_data[real_data.index >= forecast_from].values.T
realized_series = np.diff(np.log(realized_series), axis=1)  # log-differenced series
time_series = np.diff(np.log(time_series), axis=1)  # log-differenced series
print(realized_series.shape)

### AR(1) for each feature

### forecast: for T = {1, 2, 3, 4, 8, 12}
T_forecast = [1, 2, 3, 4, 8, 12]
rmsfes = np.zeros((len(T_forecast), time_series.shape[0]))  # (n_forecast_horizons, n_features)

ar1_params = []
for i in range(time_series.shape[0]):
    params = findar_p(time_series[i, :], 1)
    print(f"Feature {i}: AR(1) parameters: {params[0]} + {params[1]} * x(t-1)")
    
    
    forecast = forecast_ar1(params, T_forecast[-1], time_series[i, -1])
    ### compute forecast errors
    forecast_errors = forecast - realized_series[i, :T_forecast[-1]]
    for j, t in enumerate(T_forecast):
        rmsfe = np.sqrt(np.mean(forecast_errors[:t]**2))
        rmsfes[j, i] = rmsfe
        print(f"Feature {i}: RMSFE for T={t}: {rmsfe:.4f}")

np.savetxt("rsmfe/rmsfes_ar1.csv", rmsfes, delimiter=",", header=",".join(keys), comments="")