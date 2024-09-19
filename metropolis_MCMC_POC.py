#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:30:57 2024

@author: devere
"""

"""
Proof-of-Concept MCMC Simulation using Metropolis Algorithm:
    We'll use the Metropolis algorithm to simulate a Student's t-distribution with six 
    degrees of freedom as our target distribution.
    
    We'll use the uniform distribution as the proposal probability distribution to help 
    explore the target distribution by proposing the next state in the Markov chain.

    This simulation initializes the Markov chain arbitrarily at x = 0 and runs the Metropolis
    sampling algorith 10k times. The resulting samples are stored in a list which is then
    plotted to visualize the sample path of the Markov chain. The final histogram of the
    samples is used to show convergence to the actual target distribution.
"""

#%%
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt


#%%
# Define the target distribution as a Student's t with 6 DOF and location = 0 with 
# scale = 1
def target(x):
    return stats.t.pdf(x, df = 6)


#%% Define the proposal distribution as a uniform
def proposal(x):
    return stats.uniform.rvs(loc = x-0.5, scale = 1)


#%%
# Set then initial state arbitrarily as x = 0 and set the number of iterations to 10k
x0 = 0
n_iter = 10_000


#%%
# Initialize the Markov chain and the samples list
x = x0
samples = [x]


#%%
# Run the Metropolis algorithm to generate new samples and store them in the samples list
for i in range(n_iter):
    # Generate a proposed state from the proposal distribution
    x_proposed  = proposal(x)

    # Calculate the acceptance ratio
    acceptance_ratio = target(x_proposed) / target(x)

    # Accept or reject the proposed state
    if acceptance_ratio >= 1:
        # Accept the new sample
        x = x_proposed
    else:
        u = np.random.rand()
        # Reject the new sample
        if u < acceptance_ratio:
            x = x_proposed
            
    # Add the current state to the list of samples
    samples.append(x)
    
#%%
# Plot the sample path of the Markov chain
plt.plot(samples)
plt.xlabel('Sample Number')
plt.ylabel('Sample Value')
plt.title('Sample Path of the Markov Chain')
plt.show()

# Plot the histogram of the samples and compare it with the target distribution
plt.hist(samples, bins=50, density=True, alpha=0.5, label='MCMC Samples')
x_range = np.linspace(-5, 5, 1000)
plt.plot(x_range, target(x_range), 'r-', label='Target Distribution')
plt.xlabel('Sample Value')
plt.ylabel('Probability Density')
plt.title('MCMC Simulation of Students-T Distribution')
plt.legend()
plt.show()

#%%
# So, we can see from the graphics how we were able to use the sampling algorithm to 
# simulate a large number of samples from a Student's t-distribution while using a uniform
# distribution to guide our random walk.