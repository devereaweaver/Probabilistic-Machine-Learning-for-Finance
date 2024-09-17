#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:10:32 2024

@author: devere
"""

"""
A Probabilistic Model for Earnings Expections:
    The following code is used to compute a numerical grid approximation for the problem 
    of a company beating its earnings in the final quarter given it has beaten its earnings
    in all previous quarters. 
    
    Instead of using an MLE point estimate, we'll use a probabilistic framework to compute
    the probability distribution of the outcome with the sample data. 
    
    This involves:
        1. Specify a prior probability distribtuion that encapsulates our knowledge or 
        hypothesis about model parameters before we observe any data. We'll use 
        P(p) ~ Unif(0,1) as an uninformative prior
        
        2. Specify a likelihood function that gives us the plausibility of observing our
        in-sample data assuming any value for our parameter. We'll use 
        P(D|p) ~ Bern(p) as our likelihood function
        
        3. Using these distributions, we'll compute the likelihood, unnormalized posterior, 
        and finally the normalized posterior.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# We'll be computing this posterior numerically using grid approximation with 9 points
# instead of attempting to do so analytically by hand.

# The first thing to do is specify the number of grid points.
# Here will be from .1, .2, ... .9 as our points.
p = np.arange(0.1, 1, 0.1)


# %%
# Next, we'll need to assign the probability to each point in our grid.
# Since we're using the uniform distribution as the prior (uninformative), each point
# is equiprobable.
prior = 1 / len(p)


# %%
# Create a dataframe with the columns to store our individual calculations
earnings_beat = pd.DataFrame(
    columns=["Parameter", "Prior", "Likelihood", "Posterior*", "Posterior"]
)
earnings_beat


# %%
# Store each parameter value in the dataframe
earnings_beat["Paremater"] = p


# %%
# This cell uses a loop to compute the unnormalized posterior probability distribution
# for each value of the parameter.
for i in range(0, len(p)):
    # Store the prior probability in the dataframe [observation row, column number].
    earnings_beat.iloc[i, 1] = prior

    # Compute and store the value of the likelihood function for each parameter.
    # Recall, P(D|p) = p * p * p = p^3
    earnings_beat.iloc[i, 2] = p[i] ** 3

    # Compute and store the unnormalized posterior probability distribution.
    # (Unnormalized inverse probability rule)
    earnings_beat.iloc[i, 3] = earnings_beat.iloc[i, 1] * earnings_beat.iloc[i, 2]

    # Compute and store the normalized posterior probability distribution.
    earnings_beat["Posterior"] = earnings_beat["Posterior*"] / sum(earnings_beat["Posterior*"])

#%%
# Plot the prior and posterior probability distribution for the model parameter
plt.figure(figsize=(16,6)), plt.subplot(1,2,1), plt.ylim([0,0.5])
plt.stem(earnings_beat['Parameter'],earnings_beat['Prior'], data = earnings_beat)
plt.xlabel('Model parameter p')
plt.ylabel('Probability of parameter P(p)')
plt.title('Prior distribution of our model parameter');

plt.subplot(1,2,2), plt.ylim([0,0.5])
plt.stem(earnings_beat['Parameter'],earnings_beat['Posterior'])
plt.xlabel('Model parameter p'), plt.ylabel('Probability of parameter P(p)'), 
plt.title('Posterior distribution of our model parameter')
plt.show();


#%%
# TODO: Add the predictive prior probability distributions.






































