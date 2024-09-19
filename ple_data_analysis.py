#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:01:45 2024

@author: devereweaver

    
This file contains code to download price data for Apple, SPY and 
10-year treasury notes to compute the daily price returns for a 
linear market model. 

* Apple will be the specific stock 
* SPY will represent  our market portfolio 
* The 10-year treasury bonds will represent our risk-free asset 
    
"""

#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

#!pip install pymc -q
#%conda install pymc
import pymc as pm 
import arvix as az
az.style.use("arviz-darkgrid")

#!pip install yfinance -q
import yfinance as yf

#%%