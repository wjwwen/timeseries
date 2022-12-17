import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from scipy import stats

# %%
'''
Key takeaways:
- Detrending is crucial for time series analysis, if not correlation may be strong (inaccurate)
- Check if fluctuations are significant with CI

Autocorrelation
The degree of resemblance between a certain time series and a lagged version of itself 
over subsequent time intervals. 

ACF and PACF
- ACF: Evaluates the correlation between observations in a time series over a given range of lags
(Includes indirect correlation)

- PACF: The partial correlation for each lag is the unique correlation between the two observations 
after the intermediate correlations have been removed.
(Excludes indirect correlation)
'''

# %%
# Pearson correlation
def pearson(x, y):
    meanx = x.mean()
    meany = y.mean()
    stdx = x.std()
    stdy = y.std()
    
    return np.mean((x - meanx) * (y - meany)) / (stdx * stdy)

# Random data
x = np.random.random(1000)
y = np.random.random(1000)

# Plot - uncorrelated
fig, axs = plt.subplots(1, 1)
axs.plot(x, y, '*')
axs.set_xlabel('x')
axs.set_ylabel('y')

# Correlation close to 0
pearson(x, y)

# When a trend is added, correlation turns strong
# One of the reasons why we MUST detrend timeseries before analysis
trend = np.linspace(1, 5, 1000)
pearson(x+trend, y+trend)

# %%
# ---------------------- Auto-correlation (ACF) ----------------------

# Increasing lag --> Correlation decreases until it fluctuates around zero
def acf(x, max_lag=40):
    return np.array([1] + [pearson(x[:-i], x[i:]) for i in range(1, max_lag)])

plt.bar(range(40), acf(x, 40))
plt.xlabel('lag')
plt.ylabel('ACF')

# Check if fluctuations are significant
# CI for auto-correlation
def acf_ci(acfv, n, alpha=0.05):
    se = [1 / np.sqrt(n)]
    se.extend(np.sqrt((1+2*np.cumsum(np.power(acfv[1:-1], 2)))/n))
    se = np.array(se)
    
    se *= stats.norm.ppf(1-alpha/2.)
    return se

def plot_acf(x, lag=40, alpha=0.05):
    acf_val = acf(x, lag)
    
    plt.vlines(range(lag), 0, acf_val)
    plt.scatter(np.arange(lag), acf_val, marker='o')
    plt.xlabel('lag')
    plt.ylabel('ACF')
    
    # Determine confidence interval
    ci = acf_ci(acf_val, len(x), alpha)
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)
    
# Any correlation faling within CI can be safely ignored
plot_acf(x) 

# %%
# Example Case: GDP.csv
GDP = pd.read_csv('GDP.csv', parse_dates=['DATE'])
GDP.set_index('DATE', inplace=True)

ax=GDP.plot(legend=False)
ax.set_ylabel(r'GDP ($\$B$)')

values = GDP['GDP'].values
detrended = values[1:]-values[:-1]

# Since the series has a very strong trend, 
# auto-correlation function appears to be significant for very long periods
plot_acf(values)

# Must detrend data!
plt.plot(GDP.index[1:], detrended)
plt.xlabel('DATE')
plt.ylabel(r'QoQ Change ($\$B$)')

plot_acf(detrended)

# %%
# ---------------------- Partial Auto-correlation (PACF) ----------------------

# considers the entire time series for each lag
def rolling(x, order):
    npoints = x.shape[0]
    running = []
    
    for i in range(npoints-order+1):
        running.append(x[i:i+order])
        
    return np.array(running)

# for each lag, account for the amount of correlation 
# that has already been explained by previous lags using a
# linear model to predict Xt and Xt-1

def pacf(x, lag=40):
    y = []
    
    for i in range(3, lag + 2):
        windows = rolling(x, i)

        xt = windows[:, -1] # Current values are at the end
        xt_l = windows[:, 0] # Lagged values are at 0
        inter = windows[:, 1:-1] # Intermediate values are in between 1 and -1
        
        
        lm = LinearRegression(fit_intercept=False).fit(inter, xt)
        xt_fit = lm.predict(inter)

        lm = LinearRegression(fit_intercept=False).fit(inter, xt_l)
        xt_l_fit = lm.predict(inter)

        y.append(pearson(xt - xt_fit, xt_l - xt_l_fit))
    
    # Pad the array with the two missing values
    pacf_1 = acf(x, 2)[1]
    return np.array([1, pacf_1] +  y)

def plot_pacf(x, alpha=0.05, lag=40):
    pacf_val = pacf(x, lag)

    plt.vlines(np.arange(lag + 1), 0, pacf_val)
    plt.scatter(np.arange(lag + 1), pacf_val, marker='o')
    plt.xlabel('lag')
    plt.ylabel('PACF')
    
    # Determine confidence interval
    ci = acf_ci(pacf_val, len(x))
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)
    

plot_pacf(detrended)