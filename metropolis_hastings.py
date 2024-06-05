import numpy as np
import pandas as pd
from pprint import pprint
from utils import *
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import os
from scipy.stats import norm, uniform, lognorm, beta


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
    if not os.path.exists("plots/metropolis_hastings"):
        os.makedirs("plots/metropolis_hastings")
    plt.savefig(f"plots/metropolis_hastings/diagnostics_sampler_{sampler_num}.png")



# PROPOSAL DISTRIBUTIONS
def propose_log_normal(current, proposal_sd=0.1):
    return np.exp(np.random.normal(np.log(current), proposal_sd))


def propose_truncated_normal(current, proposal_sd=0.1, lower=0.1, upper=10):
    proposal = np.random.normal(current, proposal_sd)
    return np.clip(proposal, lower, upper)

def propose_normal(current, proposal_sd=0.1):
    return np.random.normal(current, proposal_sd)

def propose_beta(current, a=2, b=2):
    return np.random.beta(a, b) * (1 - 0) + 0

def propose_beta_adaptive(current, a_base=2, b_base=2, scale_factor=10):
    # Scale shape parameters based on the current value
    alpha = a_base + scale_factor * current
    beta_param = b_base + scale_factor * (1 - current)
    proposed_value = beta.rvs(alpha, beta_param)
    return proposed_value

def propose_uniform(current, lower=0, upper=1):
    return np.random.uniform(lower, upper)

def propose_uniform_mu_gamma(current):
    lower = current - current * 0.5
    upper = current + current * 0.5
    if lower > upper:
        lower, upper = upper, lower
    return np.random.uniform(lower, upper)

def propose_inverse_gamma(current, alpha=2, beta=2):
    return 1 / np.random.gamma(alpha, scale=1/beta)

def propose_t_distribution(current, df=2, scale=0.1):
    return current + np.random.standard_t(df) * scale

def propose_reflective_normal(current, proposal_sd=0.1, lower=None, upper=None):
    proposal = np.random.normal(current, proposal_sd)
    if lower is not None and proposal < lower:
        proposal = lower + (lower - proposal)
    if upper is not None and proposal > upper:
        proposal = upper - (proposal - upper)
    return proposal

def propose_logistic_normal(current, proposal_sd=0.1):
    from scipy.special import expit, logit
    logit_current = logit(current)
    proposal = np.random.normal(logit_current, proposal_sd)
    return expit(proposal)

def propose_exponential(current, scale=1.0):

    proposal = np.random.exponential(scale)
    proposal += current - scale 

    return proposal



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
        breakpoint()
        log_likelihood += -np.sum((yi - mu)**2) / (2 * sigma2) - np.log(2 * np.pi * sigma2)
    return log_likelihood



# SAMPLER
def metropolis_hastings(n_samples, initial_theta, proposal_funcs, prior_funcs, likelihood_func, data):
    np.random.seed(42)  # Set seed for reproducibility
    samples = np.zeros((n_samples, len(initial_theta)))
    samples[0, :] = initial_theta
    current_theta = initial_theta
    current_likelihood = likelihood_func(current_theta, data)
    current_prior_log_prob = np.sum([prior_funcs[i](current_theta[i]) for i in range(len(current_theta))])
    
    for i in range(1, n_samples):
        if i % 1000 == 0:
            print(f"Generated {i} samples")
        proposed_theta = np.array([proposal_funcs[j](current_theta[j]) for j in range(len(current_theta))])
        proposed_likelihood = likelihood_func(proposed_theta, data)
        proposed_prior_log_prob = np.sum([prior_funcs[j](proposed_theta[j]) for j in range(len(proposed_theta))])
        
        acceptance_log_prob = (proposed_likelihood + proposed_prior_log_prob) - (current_likelihood + current_prior_log_prob)
        if np.log(np.random.rand()) < acceptance_log_prob:
            current_theta = proposed_theta
            current_likelihood = proposed_likelihood
            current_prior_log_prob = proposed_prior_log_prob
        
        samples[i, :] = current_theta
    
    return samples


# Load data
raw_data = pd.read_csv("data.csv")
data = preprocess_data(raw_data)


# Calculate the sample variance for the combined gene expression values
combined_variance = raw_data[['gene1', 'gene2']].var().mean()
# Initial parameter seeds
initial_theta = [combined_variance, 0.5, raw_data.loc[raw_data['group'] == 1, 'gene1'].mean(), raw_data.loc[raw_data['group'] == 1, 'gene2'].mean(), raw_data.loc[raw_data['group'] == 2, 'gene1'].mean(), raw_data.loc[raw_data['group'] == 2, 'gene2'].mean()]

# Prior functions
prior_funcs = [
    prior_sigma2,
    prior_tau,
    prior_mu_gamma,
    prior_mu_gamma,
    prior_mu_gamma,
    prior_mu_gamma
]

# Proposal functions
proposal_combinations = [
    [propose_truncated_normal, propose_beta_adaptive, propose_normal, propose_normal, propose_normal, propose_normal],
    [propose_normal, propose_beta_adaptive, propose_normal, propose_normal, propose_normal, propose_normal],
    [propose_truncated_normal, propose_beta_adaptive, propose_t_distribution, propose_t_distribution, propose_t_distribution, propose_t_distribution],
    [propose_normal, propose_beta_adaptive, propose_t_distribution, propose_t_distribution, propose_t_distribution, propose_t_distribution],
    [propose_truncated_normal, propose_beta_adaptive, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma],
    [propose_normal, propose_beta_adaptive, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma],
    [propose_truncated_normal, propose_uniform, propose_normal, propose_normal, propose_normal, propose_normal],
    [propose_normal, propose_uniform, propose_normal, propose_normal, propose_normal, propose_normal],
    [propose_truncated_normal, propose_uniform, propose_t_distribution, propose_t_distribution, propose_t_distribution, propose_t_distribution],
    [propose_normal, propose_uniform, propose_t_distribution, propose_t_distribution, propose_t_distribution, propose_t_distribution],
    [propose_truncated_normal, propose_uniform, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma],
    [propose_normal, propose_uniform, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma, propose_uniform_mu_gamma],
    
]




# Run sampler, save diagnostics
for i, proposal_funcs in enumerate(proposal_combinations):
    print(f"Running sampler {i}")
    samples = metropolis_hastings(10000, initial_theta, proposal_funcs, prior_funcs, likelihood, data)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    for j in range(-10, 0):
        print(f"Sample {j}: {samples[j]}")
    plot_diagnostics(i, samples, list(range(0, len(initial_theta))), ['sigma^2', 'tau', 'mu1', 'mu2', 'gamma1', 'gamma2'])
