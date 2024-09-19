#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:27:11 2024

@author: devereweaver

Market Model with statsmodels:
    This is an example of building a conventional MLE-based market model using software.
    An MLE-based model differs from a probabilistic model in that the former outputs a 
    point estimate while the latter(?) is used to compute a probability distribution for 
    our parameter.
"""
# %%
!pip install yfinance
import statsmodels.api as sm
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime


# %%
# Import Financial Data

# first set start and end times to specify the time series duration
start = datetime(2017, 8, 3)
end = datetime(2022, 8, 6)

# Import historical SPY data as proxy for market portfolio
market = yf.Ticker("SPY").history(start=start, end=end)

# Import historical Apple data to use as specific stock
stock = yf.Ticker("AAPL").history(start=start, end=end)

# Import historical 10 year US treasury note as proxy for risk free rate
risk_free_rate = yf.Ticker("^TNX").history(start=start, end=end)


# %%
# Create dataframe to hold daily returns of securities by computing
# the fractional percent changes of each time series observation
# (1 = 1 period)
daily_returns = pd.DataFrame()
daily_returns["market"] = market["Close"].pct_change(1)*100
daily_returns["stock"] = stock["Close"].pct_change(1)*100

# Compute compounded daily rate based on 360 days for the calendary year
# used in the bond market to get the daily returns of the risk free asset
risk_free_rate.index = daily_returns.index
daily_returns["riskfree"] = (1 + risk_free_rate["Close"]) ** (1/360) - 1

# Note: We needed to reset the index values for the risk free rate series
# because the hours in the daily returns and risk free rate series indices
# are off by one hour, which will result in NaNs populating the dataframe

# %%
# Plot and summarize the distribution of daily returns of the marke portfolio
plt.hist(daily_returns["market"])
plt.title("Distribution of Market (SPY) Daily Returns")
plt.xlabel("Daily Percentage Returns")
plt.ylabel("Frequency")
plt.show()
print(f"Descriptive statistics of the Market's daily percentage returns:"
      f"\n{daily_returns['market'].describe()}")

# %%
# Plot and summarize the distribution of daily returns of AAPL
plt.hist(daily_returns["stock"])
plt.title("Distribution of Apple (AAPL) Daily Returns")
plt.xlabel("Daily Percentage Returns")
plt.ylabel("Frequency")
plt.show()
print(f"Descriptive statistics of the Apple's daily percentage returns:"
      f"\n{daily_returns['stock'].describe()}")

# %%
# Plot and summarize the distribution of daily returns of risk free asset
plt.hist(daily_returns["riskfree"])
plt.title("Distribution of Risk Free Asset Daily Returns")
plt.xlabel("Daily Percentage Returns")
plt.ylabel("Frequency")
plt.show()
print(f"Descriptive statistics of the Apple's daily percentage returns:"
      f"\n{daily_returns['riskfree'].describe()}")

# %%
# Examine any missing rows in the dataframe
market.index.difference(risk_free_rate.index)

# Fill rows with previou day's risk free rate daily rates if there are any missing
daily_returns = daily_returns.ffill()

# Drop the first observation because of percentage calulations result in NAs
# since there are no observations before it
daily_returns = daily_returns.dropna()

# Check dataframe for null values
daily_returns.isnull().sum()

daily_returns.head()

# %%
# Compute AAPL's Market Model based on daily excess returns
# (recall, this is in excess of the risk free asset)
y = daily_returns["stock"] - daily_returns["riskfree"]

# Compute the market's excess returns
x = daily_returns["market"] - daily_returns["riskfree"]

# Plot the execess returns
plt.scatter(x, y)

# Add the constant vector of ones to obtain the intercept
x = sm.add_constant(x)

# Use ols to find line of best fit
market_model = sm.OLS(y, x).fit()

# Plot line of best fit
plt.plot(x, x*market_model.params[0] + market_model.params["const"])
plt.title("Market Model of AAPL")
plt.xlabel("SPY Daily Excess Returns")
plt.ylabel("AAPL Daily Excess Returns")
plt.show()

# Display the values of alpha and beta of AAPL's market model
print("According to AAPL's Market Model, the security has a realized Alpha of "
      f"{round(market_model.params['const'], 2)} and a Beta of "
      f"{round(market_model.params[0],2)}.")

#%%
"""
So, after running this regression, an analyst would determine that the alpha of
Apple over that time period is 0.071% and its market risk is 1.24

This can be seen by the following statsmodels summary
"""
market_model.summary()
