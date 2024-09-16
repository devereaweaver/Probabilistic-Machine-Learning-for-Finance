#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:37:12 2024

@author: devere
"""

# %%
import pandas as pd


# %%
# Create a dataframe for bond analysis with the two outcomes as indices
bonds = pd.DataFrame(index=["Default", "No Default"])


# %%
# Define our priors where P(Default) = .10 and P(No Default) = .90
bonds["Prior"] = 0.10, 0.90


# %%
# Define the likelihood functions for observing negative ratings
# P(Negative | Default) = .70 and P(Negative | No Default) = .40
bonds["Likelihood_Negative"] = 0.70, 0.40


# %%
# Define joint probabilities of seeing a negative rating from our system depending on
# default or no default
# P(Negative | Default)P(Default) and P(Negative | No Default)P(No Default)
# (This is the value that is in the numerator of the inverse probability rule)
bonds["Joint1"] = bonds["Likelihood_Negative"] * bonds["Prior"]


# %%
# Define and compute the marginal probability (unconditional) of observing a negative
# rating (again, this is the value that is in the denominator of the inverse probability rule)
probabiility_negative_data = bonds["Joint1"].sum()


# %%
# Finally, use all these values to compute the posterior for a company defaulting given
# we've seen a negative value from the system and for a company no defaulting given we've
# seen a negative value from the system
bonds["Posterior1"] = round(bonds["Joint1"] / probabiility_negative_data, 2)


# %%
# A few days later, we can update the probability of default in a continuous manner by
# using our previous posterior as the current prior to compute the next posterior.
bonds["Joint2"] = bonds["Likelihood_Negative"] * bonds["Posterior1"]
probabiility_negative_data = bonds["Joint2"].sum()
bonds["Posterior2"] = round(bonds["Joint2"] / probabiility_negative_data, 2)

# Observe how the probability of default keeps increasing with each negative rating.
# %%
# Create a new table so that you can plot a graph with the appropriate information
table = bonds[["Prior", "Posterior1", "Posterior2"]].round(2)

# Change columns so that x axis is the number of negative ratings
table.columns = ["0", "1", "2"]

# Select the row to plot in the graph and print it.
default_row = table.iloc[0]
default_row.plot(
    figsize=(8, 8),
    grid=True,
    xlabel="Updates based on recent negative ratings",
    ylabel="Probability of default",
    title="XYZ Bonds",
)
