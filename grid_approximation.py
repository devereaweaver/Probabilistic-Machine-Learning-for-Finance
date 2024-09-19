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
earnings_beat["Parameter"] = p


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

# %%
# Plot the prior and posterior probability distribution for the model parameter
plt.figure(figsize=(16, 6)), plt.subplot(1, 2, 1), plt.ylim([0, 0.5])
plt.stem(earnings_beat["Parameter"], earnings_beat["Prior"])
plt.xlabel("Model parameter p")
plt.ylabel("Probability of parameter P(p)")
plt.title("Prior distribution of our model parameter")

plt.subplot(1, 2, 2), plt.ylim([0, 0.5])
plt.stem(earnings_beat["Parameter"], earnings_beat["Posterior"])
plt.xlabel("Model parameter p"), plt.ylabel("Probability of parameter P(p)"),
plt.title("Posterior distribution of our model parameter")
plt.show()

# This visualization shows that when we started with an uninformed prior, all points in
# the grid were equiprobable. However, once we added the information that the company has
# beat earnings expectations for the past three quarterly earnings, we see that the
# probability is greater for those points that represent a higher chance of beating earnings
# this quarter.

# This is what is meant by probabilistic models compute a probability distribution and not
# a point estimate for a parameter. Instead of computing a single point estimate that
# represents the probability of exceeding earnings, we computed the probability for multiple
# parameters. This is one crucial aspect of probabilistic modeling that makes it different
# from frequentist statistical modeling.

# I.e. it shows a probability distribution for the model parameter p BEFORE and AFTER
# training the model on in-sample data D. This is more realistic given that we always
# have incomplete information about any event.


# %%
# Now that we have computed a prior probability distribution, we need to use it for prediction
# to predict the probability of the company beating the market's expectations of its fourth
# quarter earnings estimates.
# To do so, we'll need to develop the predictive distributions of our model.

# Recall, a prior predictive distribution P(D') is the prior probability distribution of
# simulated data we expect to observe in the training data before we start training the
# model. (This needs further explaination for me)

# The posterior predictive distribution simulates posterior probability distribution of
# out-of-sample OR test data we expect to observe in the future after we have trained our
# model on the training data. (Again, this is going to need more explanation for me.)

# Just for some clarification for my sake, the posterior distribution refers to the
# distribution of the parameter, in this case p.

# The posterior predictive distribution refers to the distribution of future observations of
# data.

# So what we're doing now the, is computing the distribution of observing a y = 1 or y = 0
# in the future (beat expectations or don't), based on the posterior distribution.

# %%
# First we'll do the prior predictive which computes the probability weighted average of
# observing y = 1 using our prior probabilities as weights. This probability weighted
# average gives us the prior predictive probability of observing y = 1.
prior_predictive_1 = sum(earnings_beat["Parameter"] * earnings_beat["Prior"])

# The prior predictive of observing outcome y = 0 is just the compliment of y = 1.
prior_predictive_0 = 1 - prior_predictive_1

# Recall, we picked the uniform distribution for our parameter, thus our prior predictive
# will show that both outcomes are equally likely PRIOR TO OBSERVING ANY DATA.
print(prior_predictive_0, prior_predictive_1)


# %%
# Now, we'll compute the probability weighted average of observing y = 1 but using the
# posterior probabilites as the weights this time. This gives us the posterior predictive
# probability of observing y=1 AFTER observing in-sample data, D = {y1=1, y2=1, y3=1}
posterior_predictive_1 = sum(earnings_beat["Parameter"] * earnings_beat["Posterior"])
posterior_predictive_0 = 1 - posterior_predictive_1
print(round(posterior_predictive_0, 2), round(posterior_predictive_1, 2))

# So, after observing out in-sample data, our probabilistic model predicts that observing
# y=1 is about three times more likely than observing y=0.

# I need to review this problem again in its entirety at a later point.

# In summary, this is an example of using the grid approximation method to numerically
# compute probability distributions. It isn't too difficult to follow; however, the
# tradeoff is that this numerical approximation technique doesn't scale well when the model
# has more than a few parameters we need to estimate distributions for.
