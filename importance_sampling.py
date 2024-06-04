# Run importance sampling and save diagnostics
import numpy as np
import pandas as pd
from pprint import pprint
from utils import *
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import os
from scipy.stats import norm, uniform, lognorm



# Define prior functions
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

# Likelihood function
def likelihood(theta, data):
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

# Proposal distribution functions
def propose_log_normal():
    return lognorm(s=1).rvs()

def propose_uniform(lower=0, upper=1):
    return uniform(lower, upper-lower).rvs()

def propose_normal(mean=0, sd=1):
    return norm(mean, sd).rvs()


# PLOTTING DIAGNOSTICS
def plot_diagnostics(sampler_num, samples, parameter_indices, parameter_names):
    """
    Plots diagnostics for selected parameters in the Metropolis-Hastings samples.

    Args:
    samples (numpy.ndarray): The array of samples.
    parameter_indices (list): List of indices for the parameters to be diagnosed.
    parameter_names (list): List of names for the parameters.
    """
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
    if not os.path.exists("plots/metropolis_hastings"):
        os.makedirs("plots/metropolis_hastings")
    plt.savefig(f"plots/metropolis_hastings/diagnostics_sampler_{sampler_num}.png")

# Importance sampling function
def importance_sampling(n_samples, proposal_funcs, prior_funcs, likelihood_func, data):
    # Set seed for reproducibility
    np.random.seed(42)
    samples = np.zeros((n_samples, len(proposal_funcs)))
    weights = np.zeros(n_samples)
    
    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"Generated {i} samples")
        
        sample = np.array([proposal_funcs[k]() for k in range(len(proposal_funcs))])
        log_prior = np.sum([prior_funcs[k](sample[k]) for k in range(len(sample))])
        log_likelihood = likelihood_func(sample, data)
        log_proposal = np.sum([np.log(proposal_funcs[k](sample[k])) for k in range(len(sample))])
        
        log_weight = log_prior + log_likelihood - log_proposal
        samples[j, :] = sample
        weights[j] = np.exp(log_weight)
    
    weights /= np.sum(weights)
    
    return samples, weights

# Load data
raw_data = pd.read_csv("data.csv")
data = preprocess_data(raw_data)

# Calculate the sample variance for the combined gene expression values
combined_variance = raw_data[['gene1', 'gene2']].var().mean()

# Initial parameter seeds
initial_theta = [combined_variance, 0.5, 
                 raw_data.loc[raw_data['group'] == 1, 'gene1'].mean(), 
                 raw_data.loc[raw_data['group'] == 1, 'gene2'].mean(), 
                 raw_data.loc[raw_data['group'] == 2, 'gene1'].mean(), 
                 raw_data.loc[raw_data['group'] == 2, 'gene2'].mean()]

# Prior functions
prior_funcs = [
    prior_sigma2,
    prior_tau,
    prior_mu_gamma,
    prior_mu_gamma,
    prior_mu_gamma,
    prior_mu_gamma
]

# Proposal functions combinations
proposal_combinations = [
    [propose_log_normal, lambda: propose_uniform(0, 1), lambda: propose_normal(initial_theta[2], 1), lambda: propose_normal(initial_theta[3], 1), lambda: propose_normal(initial_theta[4], 1), lambda: propose_normal(initial_theta[5], 1)],
    [lambda: propose_normal(initial_theta[0], 1), lambda: propose_uniform(0, 1), lambda: propose_normal(initial_theta[2], 1), lambda: propose_normal(initial_theta[3], 1), lambda: propose_normal(initial_theta[4], 1), lambda: propose_normal(initial_theta[5], 1)],
    [propose_log_normal, lambda: propose_uniform(0, 1), lambda: propose_normal(initial_theta[2], 2), lambda: propose_normal(initial_theta[3], 2), lambda: propose_normal(initial_theta[4], 2), lambda: propose_normal(initial_theta[5], 2)],
    [lambda: propose_normal(initial_theta[0], 1), lambda: propose_uniform(0, 1), lambda: propose_normal(initial_theta[2], 2), lambda: propose_normal(initial_theta[3], 2), lambda: propose_normal(initial_theta[4], 2), lambda: propose_normal(initial_theta[5], 2)],
    [propose_log_normal, lambda: propose_uniform(0, 1), lambda: propose_uniform(initial_theta[2]-0.5*initial_theta[2], initial_theta[2]+0.5*initial_theta[2]), lambda: propose_uniform(initial_theta[3]-0.5*initial_theta[3], initial_theta[3]+0.5*initial_theta[3]), lambda: propose_uniform(initial_theta[4]-0.5*initial_theta[4], initial_theta[4]+0.5*initial_theta[4]), lambda: propose_uniform(initial_theta[5]-0.5*initial_theta[5], initial_theta[5]+0.5*initial_theta[5])],
    [lambda: propose_normal(initial_theta[0], 1), lambda: propose_uniform(0, 1), lambda: propose_uniform(initial_theta[2]-0.5*initial_theta[2], initial_theta[2]+0.5*initial_theta[2]), lambda: propose_uniform(initial_theta[3]-0.5*initial_theta[3], initial_theta[3]+0.5*initial_theta[3]), lambda: propose_uniform(initial_theta[4]-0.5*initial_theta[4], initial_theta[4]+0.5*initial_theta[4]), lambda: propose_uniform(initial_theta[5]-0.5*initial_theta[5], initial_theta[5]+0.5*initial_theta[5])]
]


results = []
for num, proposal_funcs in enumerate(proposal_combinations):
    print(f"Running sampler {num}")
    results.append(importance_sampling(10000, proposal_funcs, prior_funcs, likelihood, data))

for i, (samples, weights) in enumerate(results):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    for j in range(-10, 0):
        print(f"Sampler {i}, Sample {j}: {samples[j]} with weight {weights[j]}")
    # Save or plot diagnostics as needed
    plot_diagnostics(i, samples, [0, 1, 2, 3, 4, 5], ['sigma2', 'tau', 'mu1', 'mu2', 'gamma1', 'gamma2'])
