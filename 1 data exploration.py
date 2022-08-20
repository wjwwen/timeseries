# 1 -- DATA EXPLORATION
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# %% 
# Data: Infant Mortality
ILI = pd.read_csv('/Users/jingwen/Desktop/CDC.csv')
ILI.head()

ILI['date'] = ILI['Year']+ILI['Week']/52

ILI.head()

ILI.plot(x='date', 
         y=['Percent of Deaths Due to Pneumonia and Influenza', 
            'Expected', 
            'Threshold'])
ax = plt.gca()
ax.legend(['Mortality', 'Expected', 'Threshold'])
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')

# %% 
# Data: Dow-Jones Industrial Average
DJIA = pd.read_csv('/Users/jingwen/Desktop/DJIA.csv', parse_dates=['DATE'], na_values='.')

DJIA.info()
DJIA.set_index('DATE', inplace=True)

ax = DJIA['2017':'2018'].plot(legend=False)
ax.set_ylabel('DJIA')
ax.set_xlabel('Date')

# %% 
# Data: Airline Passengers
airline = pd.read_csv('/Users/jingwen/Desktop/international-airline-passengers.csv', sep=';')
airline.head()

airline['Month'] = pd.to_datetime(airline['Month']+'-01')
airline.set_index('Month', inplace=True)
airline.head()

ax = airline.plot(legend=False)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')

# %%
# Stationarity
x = np.linspace(0, np.pi*10, 360)
y = np.sin(x)

fig, axs = plt.subplots(2, 2, figsize=(18, 18))
axs[0][0].plot(x, y)
axs[0][0].set_title('Stationary series')
axs[0][0].set_xlabel('time')
axs[0][0].set_ylabel('Amplitude')

axs[0][1].plot(x, y+x/10)
axs[0][1].set_title('Changing mean')
axs[0][1].set_xlabel('time')
axs[0][1].set_ylabel('Amplitude')

axs[1][0].plot(x, y*x/10)
axs[1][0].set_title('Changing Variance')
axs[1][0].set_xlabel('time')
axs[1][0].set_ylabel('Amplitude')

axs[1][1].plot(np.sin(x+x*x/30))
axs[1][1].set_title('Changing Co-variance')
axs[1][1].set_xlabel('time')
axs[1][1].set_ylabel('Amplitude')

plt.tight_layout()

# %%
# Trends
fig, axs = plt.subplots(1, 2, figsize=(18, 9))
axs[0].plot(x, y+x/10)
axs[0].set_title('Additive Trend')
axs[0].set_xlabel('time')
axs[0].set_ylabel('Amplitude')


axs[1].plot(x, y*x/10)
axs[1].set_title('Multiplicative Trend')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Amplitude')

plt.tight_layout()