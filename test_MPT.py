#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:36:03 2024

@author: devereweaver
"""

"""
test_MPT.py - we'll test the fundamental tenet of MPT that asset price
returns are normally distributed and time invariant using 30 years worth
of S&P 500 price data and computing its daily returns, skewness, and 
kurtosis
"""

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#plt.style.use("seaborn")

# Import over 30 years of S&P 500 ("SPY") price data
start = datetime(1993, 2, 1)
end = datetime(2022, 10, 15)
equity = yf.Ticker("SPY").history(start = start, end = end)

# Use SPY's closing prices to compute its daily returns
# Remove NaNs from data
equity["Returns"] = equity["Close"].pct_change(1)*100
equity = equity.dropna()

# Visualize and summarize SPY's daily price returns 
# Compute its skewness and kurtosis
plt.hist(equity["Returns"])
plt.title("Distribution of S&P 500 Daily Percentange Returns Over the Past 30 Years") 
plt.xlabel("Daily Percentage Returns")
plt.ylabel("Frequency")

print(f"Descriptive statistics of S&P 500 percentage returns:\n {equity['Returns'].describe().round(2)}")

print(f"\nThe skewness of S&P 500 returns is: {equity['Returns'].skew().round(2)}")

print(f"\nThe kurtosis of S&P 500 returns is: {equity['Returns'].kurtosis().round(2)}")



















