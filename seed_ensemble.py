import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sys import argv
from os import path, listdir
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR

### load all posterios for seeds

def select_optimal_lag_lengths(x, start_from=0, max_lag=None, min_lag=1):
    """
    Select optimal lag lengths for AR and VAR models using rolling
    one-step-ahead re-estimation and average AIC/BIC.

    Parameters:
    -----------
    x : ndarray
        2D array of shape (n_features, n_samples)
    max_lag : int, optional
        Largest lag order to evaluate. Defaults to a small data-driven bound.
    min_lag : int, optional
        Smallest lag order to evaluate.

    Returns:
    --------
    dict
        Nested dictionary with the lag that minimizes AIC and BIC for AR and VAR.
    """
    if x.ndim != 2:
        raise ValueError("x must be 2-dimensional")

    n_features, n_samples = x.shape
    if n_samples <= min_lag:
        raise ValueError("x must contain more samples than min_lag")

    if max_lag is None:
        max_lag = max(min_lag, min(10, n_samples // 3))

    if max_lag < min_lag:
        raise ValueError("max_lag must be greater than or equal to min_lag")

    lag_candidates = range(min_lag, min(max_lag, n_samples - 1) + 1)

    def average_criteria(scores):
        valid_scores = [score for score in scores if np.isfinite(score)]
        if not valid_scores:
            return np.inf
        return float(np.mean(valid_scores))

    def safe_information_criteria(result):
        try:
            return result.aic, result.bic
        except Exception:
            return None, None

    ar_aic_scores = {}
    ar_bic_scores = {}
    for p in lag_candidates:
        feature_aic = []
        feature_bic = []
        for feature_idx in range(n_features):
            series = x[feature_idx, start_from:]
            rolling_aic = []
            rolling_bic = []
            for end_idx in range(p + 2, n_samples + 1):
                try:
                    result = AutoReg(series[:end_idx], lags=p).fit()
                except Exception:
                    continue
                aic, bic = safe_information_criteria(result)
                if aic is None or bic is None:
                    continue
                rolling_aic.append(aic)
                rolling_bic.append(bic)
            feature_aic.append(average_criteria(rolling_aic))
            feature_bic.append(average_criteria(rolling_bic))
        ar_aic_scores[p] = average_criteria(feature_aic)
        ar_bic_scores[p] = average_criteria(feature_bic)

    var_aic_scores = {}
    var_bic_scores = {}
    for p in lag_candidates:
        rolling_aic = []
        rolling_bic = []
        for end_idx in range(p + 2, n_samples + 1):
            try:
                result = VAR(x[:, start_from:end_idx].T).fit(p)
            except Exception:
                continue
            aic, bic = safe_information_criteria(result)
            if aic is None or bic is None:
                continue
            rolling_aic.append(aic)
            rolling_bic.append(bic)
        var_aic_scores[p] = average_criteria(rolling_aic)
        var_bic_scores[p] = average_criteria(rolling_bic)

    return {
        "ar": {
            "aic": min(ar_aic_scores, key=ar_aic_scores.get),
            "bic": min(ar_bic_scores, key=ar_bic_scores.get),
            "scores": {
                "aic": ar_aic_scores,
                "bic": ar_bic_scores,
            },
        },
        "var": {
            "aic": min(var_aic_scores, key=var_aic_scores.get),
            "bic": min(var_bic_scores, key=var_bic_scores.get),
            "scores": {
                "aic": var_aic_scores,
                "bic": var_bic_scores,
            },
        },
    }


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

print(select_optimal_lag_lengths(real_data.values.T, start_from=10, max_lag=10, min_lag=1))