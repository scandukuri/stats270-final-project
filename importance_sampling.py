import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, invgamma
from statsmodels.graphics.tsaplots import plot_acf
import os

# PLOTTING DIAGNOSTICS
def plot_diagnostics(sampler_num, samples, weights, parameter_indices, parameter_names):
    num_params = len(parameter_indices)
    fig, axes = plt.subplots(num_params, 3, figsize=(15, 1.5 * num_params))

    for i, idx in enumerate(parameter_indices):
        ax1 = axes[i][0]
        ax1.plot(samples[:, idx])
        ax1.set_title(f'Trace Plot for sampler {sampler_num}, parameters: {parameter_names[i]}')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel(parameter_names[i])

        ax2 = axes[i][1]
        plot_acf(weights * samples[:, idx], ax=ax2, lags=50)
        ax2.set_title(f'Weighted Autocorrelation for sampler {sampler_num}, parameters: {parameter_names[i]}')

        ax3 = axes[i][2]
        ax3.hist(samples[:, idx], bins=20, weights=weights, color='blue', edgecolor='black', alpha=0.7)
        ax3.set_title(f'Weighted Density Plot for Sampler {sampler_num}, Parameter: {parameter_names[i]}')
        ax3.set_xlabel(parameter_names[i])
        ax3.set_ylabel('Density')

    fig.tight_layout()
    if not os.path.exists('plots/importance_sampling'):
        os.makedirs('plots/importance_sampling')
    plt.savefig(f'plots/importance_sampling/diagnostics_sampler_{sampler_num}.png')

# Proposal Distributions and PDFs
def propose_log_normal(current, proposal_sd=0.1):
    return np.exp(np.random.normal(np.log(current), proposal_sd))

def pdf_log_normal(x, current, proposal_sd=0.1):
    return norm.pdf(np.log(x), loc=np.log(current), scale=proposal_sd) / x

def propose_beta(current, a=2, b=2):
    return np.random.beta(a, b)

def pdf_beta(x, current, a=2, b=2):
    return beta.pdf(x, a, b)

def propose_normal(current, proposal_sd=0.1):
    return np.random.normal(current, proposal_sd)

def pdf_normal(x, current, proposal_sd=0.1):
    return norm.pdf(x, loc=current, scale=proposal_sd)

# Prior Distributions
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

# Likelihood Function
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
        
        log_likelihood += -np.sum((yi - mu) ** 2) / (2 * sigma2) - np.log(2 * np.pi * sigma2)
    return log_likelihood

# Importance Sampling
def importance_sampling(n_samples, initial_theta, proposal_funcs, proposal_pdfs, prior_funcs, likelihood_func, data):
    samples = np.zeros((n_samples, len(initial_theta)))
    weights = np.zeros(n_samples)
    current_theta = np.array(initial_theta)

    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"Generated {i} samples")
        proposed_theta = np.array([func(current) for func, current in zip(proposal_funcs, current_theta)])
        samples[i, :] = proposed_theta
        proposal_density = np.sum([pdf(theta, current) for pdf, theta, current in zip(proposal_pdfs, proposed_theta, current_theta)])
        prior_prob = np.sum([prior(theta) for prior, theta in zip(prior_funcs, proposed_theta)])
        likelihood = likelihood_func(proposed_theta, data)
        weights[i] = np.exp(likelihood + prior_prob - proposal_density)
        current_theta = proposed_theta

    weights /= np.sum(weights)

    return samples, weights

# Load data
raw_data = pd.read_csv("data.csv")
# Placeholder for preprocess_data function
data = [(raw_data.loc[i, ['gene1', 'gene2']].values, raw_data.loc[i, 'group']) for i in range(len(raw_data))]

# Initial guesses
initial_theta = [1, 0.5, np.mean(raw_data['gene1']), np.mean(raw_data['gene2']), np.mean(raw_data['gene1']), np.mean(raw_data['gene2'])]

# Example of running the sampler
proposal_funcs = [
    propose_log_normal,
    propose_beta,
    propose_normal,
    propose_normal,
    propose_normal,
    propose_normal
]

proposal_pdfs = [
    pdf_log_normal,
    pdf_beta,
    pdf_normal,
    pdf_normal,
    pdf_normal,
    pdf_normal
]

prior_funcs = [
    prior_sigma2,
    prior_tau,
    prior_mu_gamma,
    prior_mu_gamma,
    prior_mu_gamma,
    prior_mu_gamma
]

samples, weights = importance_sampling(10000, initial_theta, proposal_funcs, proposal_pdfs, prior_funcs, likelihood, data)
plot_diagnostics(0, samples, weights, list(range(0, len(initial_theta))), ['sigma^2', 'tau', 'mu1', 'mu2', 'gamma1', 'gamma2'])
