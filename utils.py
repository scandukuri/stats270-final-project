import pandas as pd
import numpy as np

def preprocess_data(raw_data: pd.DataFrame) -> list:
    # Load data
    data = []
    for i, row in raw_data.iterrows():
        data.append(((row[1], row[2]), int(row[0])))
    return data


def compute_weighted_samples(samples, weights):
    # Compute the weighted average of all samples
    weighted_samples = np.average(samples, weights=weights, axis=0)
    return weighted_samples