import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import os
from scipy.stats import norm, invgamma, uniform
from utils import *

# PLOTTING DIAGNOSTICS
def plot_diagnostics(sampler_num, samples, parameter_indices, parameter_names):
    num_params = len(parameter_indices)
    fig, axes = plt.subplots(num_params, 3, figsize=(15, 1.5*num_params))

    for i, idx in enumerate(parameter_indices):
        # Trace
        ax1 = axes[i][0]
        ax1.plot(samples[:, idx])
        ax1.set_title(f'Trace Plot for sampler {sampler_num}, parameters: {parameter_names[i]}')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel(parameter_names[i])

        # Autocorrelation
        ax2 = axes[i][1]
        plot_acf(samples[:, idx], ax=ax2, lags=50)
        ax2.set_title(f'Autocorrelation for sampler {sampler_num}, parameters: {parameter_names[i]}')
        
        # Density
        ax3 = axes[i][2]
        ax3.hist(samples[:, idx], bins=20, color='blue', edgecolor='black', alpha=0.7)
        ax3.set_title(f'Density Plot for Sampler {sampler_num}, Parameter: {parameter_names[i]}')
        ax3.set_xlabel(parameter_names[i])
        ax3.set_ylabel('Density')

    fig.tight_layout() 
    if not os.path.exists("plots/gibbs_sampling"):
        os.makedirs("plots/gibbs_sampling")
    plt.savefig(f"plots/gibbs_sampling/diagnostics_sampler_{sampler_num}.png")

# PRIOR DISTRIBUTIONS
def prior_sigma2(sigma2):
    if 0.1 <= sigma2 <= 10:
        return -np.log(sigma2)
    return -np.inf

def prior_tau(tau):
    if 0 <= tau <= 1:
        return 0
    return -np.inf

def prior_mu_gamma(value):
    return 0

# LIKELIHOOD FUNCTION
def likelihood(theta, data, t):
    sigma2, tau, mu1, mu2, gamma1, gamma2 = theta
    log_likelihood = 0
    for yi, ti in data:
        if ti == 1:
            mu = np.array([mu1, mu2])
        elif ti == 2:
            mu = np.array([gamma1, gamma2])
        elif ti == 3:
            mu = 0.5 * np.array([mu1, mu2]) + 0.5 * np.array([gamma1, gamma2])
        elif ti == 4:
            mu = tau * np.array([mu1, mu2]) + (1 - tau) * np.array([gamma1, gamma2])
        
        log_likelihood += -np.sum((yi - mu)**2) / (2 * sigma2) - np.log(2 * np.pi * sigma2)
    return log_likelihood

# CONDITIONAL SAMPLERS
def sample_sigma2(data, mu, gamma, tau, t):
    n = len(data)  # number of data points
    residual_sum = 0
    for yi, ti in data:
        if ti == 1:
            mu_i = mu
        elif ti == 2:
            mu_i = gamma
        elif ti == 3:
            mu_i = 0.5 * mu + 0.5 * gamma
        elif ti == 4:
            mu_i = tau * mu + (1 - tau) * gamma
        residual_sum += np.sum((yi - mu_i) ** 2)
    
    alpha = n / 2
    beta = residual_sum / 2
    return invgamma.rvs(alpha, scale=beta)


def sample_tau(data, mu, gamma, sigma2, t):
    group_4_data = np.array([yi for yi, ti in data if ti == 4])  # Filter and convert to numpy array
    n_group_4 = group_4_data.shape[0]
    
    mu_diff = mu - gamma
    numerator = np.sum(group_4_data @ mu_diff / sigma2)
    denominator = np.sum(mu_diff ** 2 / sigma2)
    mean_tau = numerator / denominator
    variance_tau = 1 / denominator
    
    # Ensure tau is within the [0, 1] range
    tau_sample = np.random.normal(mean_tau, np.sqrt(variance_tau))
    print(tau_sample)
    print(np.clip(tau_sample, 0, 1))
    return np.clip(tau_sample, 0, 1)


def sample_mu(data, gamma, sigma2, tau, t):
    data_array = np.array([yi for yi, ti in data])  # Extract only the data points
    mu = np.zeros(data_array.shape[1])
    for i in range(data_array.shape[1]):
        numerator = 0
        denominator = 0
        for yi, ti in data:
            if ti == 1:
                numerator += yi[i]
                denominator += 1
            elif ti == 3:
                numerator += 0.5 * yi[i] - 0.5 * gamma[i]
                denominator += 0.5
            elif ti == 4:
                numerator += tau * yi[i] - tau * gamma[i]
                denominator += tau
        variance = sigma2 / denominator
        mean = numerator / denominator
        mu[i] = np.random.normal(mean, np.sqrt(variance))
    return mu


def sample_gamma(data, mu, sigma2, tau, t):
    data_array = np.array([yi for yi, ti in data])  # Extract only the data points
    gamma = np.zeros(data_array.shape[1])
    for i in range(data_array.shape[1]):
        numerator = 0
        denominator = 0
        for yi, ti in data:
            if ti == 2:
                numerator += yi[i]
                denominator += 1
            elif ti == 3:
                numerator += 0.5 * yi[i] - 0.5 * mu[i]
                denominator += 0.5
            elif ti == 4:
                numerator += (1 - tau) * yi[i] - (1 - tau) * mu[i]
                denominator += (1 - tau)
        variance = sigma2 / denominator
        mean = numerator / denominator
        gamma[i] = np.random.normal(mean, np.sqrt(variance))
    return gamma

# SAMPLER
def gibbs_sampling(n_samples, initial_theta, data, t):
    np.random.seed(42)  # Set seed for reproducibility
    samples = np.zeros((n_samples, len(initial_theta)))
    samples[0, :] = initial_theta
    current_theta = initial_theta
    
    for i in range(1, n_samples):
        if i % 1000 == 0:
            print(f"Generated {i} samples")
        
        sigma2 = sample_sigma2(data, np.array(current_theta[2:4]), np.array(current_theta[4:]), current_theta[1], t)
        tau = sample_tau(data, np.array(current_theta[2:4]), np.array(current_theta[4:]), sigma2, t)
        mu = sample_mu(data, np.array(current_theta[4:]), sigma2, tau, t)
        gamma = sample_gamma(data, mu, sigma2, tau, t)
        
        current_theta = [sigma2, tau, mu[0], mu[1], gamma[0], gamma[1]]
        samples[i, :] = current_theta
    
    return samples

# Load data
raw_data = pd.read_csv("data.csv")
data = preprocess_data(raw_data)
t = raw_data['group'].values

# Calculate the sample variance for the combined gene expression values
combined_variance = raw_data[['gene1', 'gene2']].var().mean()

# Initial parameter seeds
initial_theta = [combined_variance, 0.5, 
                 raw_data.loc[raw_data['group'] == 1, 'gene1'].mean(), 
                 raw_data.loc[raw_data['group'] == 1, 'gene2'].mean(), 
                 raw_data.loc[raw_data['group'] == 2, 'gene1'].mean(), 
                 raw_data.loc[raw_data['group'] == 2, 'gene2'].mean()]

# Run Gibbs sampler and save diagnostics
print("Running Gibbs sampler")
samples = gibbs_sampling(10000, initial_theta, data, t)
plot_diagnostics(0, samples, list(range(0, len(initial_theta))), ['sigma^2', 'tau', 'mu1', 'mu2', 'gamma1', 'gamma2'])

# Save samples
np.save("gibbs_samples.npy", samples)
