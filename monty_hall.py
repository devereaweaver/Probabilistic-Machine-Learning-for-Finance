#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:25:50 2024

@author: devere
"""

"""
Monte Carlo Simulation (MCS) samples randomly from probability 
distributions to generate numerous probable scenarios of a system
whose outcomes are uncertain. 

The following MCS code shows how swtiching doors is the optimal 
betting strategy for this game if played many times. 
"""

import random 
import matplotlib.pyplot as plt

# Number of iterations per simulation (we'll do four sims)
number_of_iterations = [10, 100, 1000, 10000]

# Create our figure and axis objects for creating a 2x2 plot
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 6))

# iterate over the iterations list along with its index
for i, number_of_iterations in enumerate(number_of_iterations):
    # List to store results
    stay_results = []
    switch_results = []
    
    # Collect our results 
    for j in range(number_of_iterations):
        doors = ["door 1", "door 2", "door 3"]

        # Random selection of door to place the car
        car_door = random.choice(doors)
        
        # You select a door at random
        your_door = random.choice(doors)

        # Monty can only select the door that doesn't have the 
        # car and one that you haven't chosen (create a list from 
        # the set containing doors minus the set containing your door
        # and the car door, it's the first guy in the list)
        monty_door = list(set(doors) - set([car_door, your_door]))[0]

        # The door that Monty doesn't open and the one you have not
        # chosen initially (the remaining door)
        switch_door = list(set(doors) - set([monty_door, your_door]))[0]

        # Result if you stay with your original choice and it has the 
        # car behind it
        stay_results.append(your_door == car_door)
        
        # Result if you switch doors and it has the car behind it
        switch_results.append(switch_door == car_door)

    # Probability of winning the car if you stay
    probability_staying = sum(stay_results) / number_of_iterations
    
    # Probability of winning the car if you switch doors
    probability_switching = sum(switch_results) / number_of_iterations


    # Let's plot the results
    ax = axs[i // 2, i % 2]


    # Plot the probabilities as a bar graph
    ax.bar(['stay', 'switch'], [probability_staying, probability_switching],
    color=['blue', 'green'], alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Probability of Winning')
    ax.set_title('After {} Simulations'.format(number_of_iterations))
    ax.set_ylim([0, 1])

    # Add probability values on the bars
    ax.text(-0.05, probability_staying + 0.05, '{:.2f}'
    .format(probability_staying), ha='left', va='center', fontsize=10)
    ax.text(0.95, probability_switching + 0.05, '{:.2f}'
    .format(probability_switching), ha='right', va='center', fontsize=10)

plt.tight_layout()
plt.show()



