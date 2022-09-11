import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LinearRegression

# %%
# Coin flips
def flip_coin(n_coins, n_times, p=0.5):
    return 2*(np.random.random((n_times, n_coins))<p)-1

steps = flip_coin(3, 1000)

steps[:10]

position = steps.cumsum(axis=0)

plt.plot(np.arange(1000), position.T[0], label='Run 1')
plt.plot(np.arange(1000), position.T[1], label='Run 2')
plt.plot(np.arange(1000), position.T[2], label='Run 3')
plt.xlabel('steps')
plt.ylabel('position')
plt.legend()

plt.plot(np.arange(50), steps.T[0][:50])
plt.xlabel('steps')
plt.ylabel(r'$\epsilon_i$')

# If the coin is biased then we can observe significant changes in the random walk
plt.plot(np.arange(1000), flip_coin(1, 1000, 0.5).cumsum(), label='p=0.50')
plt.plot(np.arange(1000), flip_coin(1, 1000, 0.525).cumsum(), label='p=0.525')
plt.plot(np.arange(1000), flip_coin(1, 1000, 0.55).cumsum(), label='p=0.55')
plt.xlabel('steps')
plt.ylabel('position')
plt.legend()

# %%
# Dickey-Fuller
# Test for stationarity relies on a statistical test for a unit root. 
# We modify our random walk simulation to take the parameter into account

def position_rho(steps, rho):
    position = steps.astype('float').copy()

    for i in range(1, steps.shape[0]):
        position[i] = rho*position[i-1]+steps[i]
        
    return position

# By varying, we can interpolate between stationary and non-stationary behaviors
n_steps = 4
rho_lst = np.linspace(0, 1, n_steps)
steps = flip_coin(1, 1000)

fig, axs = plt.subplots(n_steps, 1)

for i, rho in enumerate(rho_lst):
    position = position_rho(steps, rho)
    axs[i].plot(position)
    axs[i].set_title(r'$\rho=%1.2f$' % rho)

fig.tight_layout()

# The Dickey-Fuller test uses the first differences so we reintroduce the differentiate function defined before
def differentiate(values, d=1):
    # First value is required so that we can recover the original values with np.cumsum
    x = np.concatenate([[values[0]], values[1:]-values[:-1]])

    if d == 1:
        return x
    else:    
        return differentiate(x, d - 1)
    
# A simple version of the test simply returns the slope of the fit
def dftest(values):
    N = len(values)
    diff = differentiate(values)

    lm = LinearRegression()
    lm.fit(values[:-1], diff[1:])
    
    return lm.coef_

# The more different from zero, the more strongly we can be sure that the process is stationary
position = position_rho(steps, 0)
dftest(position)

position = position_rho(steps, 0.8)
dftest(position)

position = position_rho(steps, 1.0)
dftest(position)
