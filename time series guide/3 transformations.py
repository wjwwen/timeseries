import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import scipy
from scipy import stats

# %%
# DIFFERENTIATE to remove trend

DJIA = pd.read_csv('/Users/jingwen/Desktop/DJIA.csv', parse_dates=['DATE'], na_values='.').dropna()
DJIA.plot(x='DATE', legend=False)
ax = plt.gca()
ax.set_ylabel('DJIA')
ax.set_xlabel('Date')

def differentiate(values, d=1):
    # First value is required so that we can recover the original values with np.cumsum
    x = np.concatenate([[values[0]], values[1:]-values[:-1]])

    if d == 1:
        return x
    else:    
        return differentiate(x, d - 1)

values = DJIA['DJIA'].values
differences = differentiate(values)

plt.plot(DJIA['DATE'].iloc[1:], differences[1:])
plt.xlabel('Date')
plt.ylabel('Differences')

# To recover the original data, integrate the differenced points. 
# The differentiate function, included the first value of the original data in the output to make this possible.
def integrate(values, d=1):
    x = np.cumsum(values)
    
    if d == 1:
        return x
    else:
        return integrate(x, d-1)

rebuilt = integrate(differences)

# %%
np.mean(rebuilt-values)

# %%
# WINDOWING
# Calculate running values of some quantity. 
# Windowing: return the proper element at each step.

def rolling(x, order):
    npoints = x.shape[0]
    running = []
    
    for i in range(npoints-order+1):
        running.append(x[i:i+order])
        
    return np.array(running)

values = np.arange(11)
values
rolling(values, 6)
rolling(values, 2)
rolling(values, 2).mean(axis=1)
rolling(values, 2).max(axis=1)

# %%
# EXPONENTIAL SMOOTHING
# Smoothing noise. 
def ES(values, alpha= 0.05):
    N = len(values)
    S = [values[0]*alpha]
    
    for i in range(1, N):
        S.append(alpha*values[i]+(1-alpha)*S[-1])
        
    return np.array(S)

# the smaller the value of alpha, the smoother (less noisy) the result
smooth = []
smooth.append(ES(differences[1:], 0.01))
smooth.append(ES(differences[1:], 0.1))
smooth.append(ES(differences[1:], 0.5))

plt.plot(DJIA['DATE'].iloc[1:100], differences[1:100], label='Differences')
plt.plot(DJIA['DATE'].iloc[1:100], smooth[2][:99], label=r'$\alpha=0.5$')
plt.plot(DJIA['DATE'].iloc[1:100], smooth[1][:99], label=r'$\alpha=0.1$')
plt.plot(DJIA['DATE'].iloc[1:100], smooth[0][:99], label=r'$\alpha=0.01$')
plt.xlabel('Date')
plt.ylabel('Differences')
plt.legend()

# %% 
# Missing Data

# Step 1: Generate a dataset with missing values
x = np.linspace(-np.pi, np.pi, 100)
y = np.cos(x)
y_missing = y.copy()
y_missing[40:55] = np.nan

# Cosine function with several missing values at the peak
plt.plot(x, y, '*')
plt.plot(x, y_missing)


# Most common strategy is to simply keep the last known 'good' value and use it to fill in the missing data points. 
# This approach is unable to deal with missing values at the beginning of the dataset.
def ffill(y):
    y0 = y.copy()
    N = len(y0)
    
    current = None
    for i in range(1, N):
        if np.isnan(y0[i]):
            y0[i] = current
        else:
            current = y0[i]

    return y0

# Naturally, the opposite approach is also common where we use the next good value. 
# Easily handle the missing initial values but can do nothing about any values lost at the end of the time series
def bfill(y):
    y0 = y.copy()
    N = len(y0)
    
    current = None
    for i in range(N-1, 0, -1):
        if np.isnan(y0[i]):
            y0[i] = current
        else:
            current = y0[i]
    
    return y0


# Back-fill and Forward-fill are simple but powerful approachs to deal with missing data. 
# One common approach is to interpolate between the previous and the next value and connecting them with a straight line.
def interpolate(y):
    y0 = y.copy()
    N = len(y0)
    
    pos = 0
    while pos < N:
        if np.isnan(y0[pos]):
            count = 0
            
            while np.isnan(y0[pos+count]):
                count += 1
            
            current = y0[pos-1]
            future = y0[pos+count]
            slope = (future-current)/count
            
            y0[pos:pos+count] = current + np.arange(1, count+1)*slope
            
            pos += count
        else:
            pos += 1
            
    return y0

y_bfill = bfill(y_missing)
y_ffill = ffill(y_missing)
y_inter = interpolate(y_missing)

plt.plot(x, y_bfill, label='back fill')
plt.plot(x, y_ffill, label='forward fill')
plt.plot(x, y_inter, label='interpolate')
plt.plot(x, y_missing, label='Data')
plt.legend()

# %%
# RESAMPLING

mapping = DJIA['DATE'].dt.year
values = DJIA['DJIA'].values

def groupBy(values, mapping, func = np.mean):
    agg = {}
    pos = {}
    
    for i in range(values.shape[0]):
        key = mapping.iloc[i]
        
        if key not in agg:
            agg[key] = []
        
        pos[key] = i
        
        if not np.isnan(values[i]):
            agg[key].append(values[i])
        
    order = sorted(agg.keys())
    
    if func is not None:
        for key in agg:
            agg[key] = func(np.array(agg[key]).astype('float'))
            
    return agg, pos

agg, pos = groupBy(values, mapping, np.mean)

agg

aggregated = []

for key in pos:
    aggregated.append([pos[key], agg[key]])

aggregated = np.array(aggregated)

aggregated

plt.plot(DJIA['DATE'], DJIA['DJIA'])
ax = plt.gca()
ax.plot(DJIA.set_index('DATE').index[aggregated.T[0].astype('int')], aggregated.T[1])
ax.plot(DJIA.set_index('DATE').index[aggregated.T[0].astype('int')], aggregated.T[1], 'ro', markersize=10,)
ax.set_ylabel('DJIA')
ax.set_xlabel('Date')

# %%
# JACKKNIFE ESTIMATORS
# Used to estimate the variance and bias of a large population. 

def jackknife(x, func, variance = False):
    N = len(x)
    pos = np.arange(N)
    values = [func(x[pos != i]) for i in pos]
    jack = np.sum(values)/N
    
    if variance:
        values = [np.power(func(x[pos != i]) - jack, 2.0) for i in pos]
        var = (N-1)/N * np.sum(values)
        return jack, var
    else:
        return jack
    
x = np.random.normal(0, 2, 100)
print(x.std())
jackknife(x, np.std, True)

# %%
# BOOTSTRAPPING
# Sample (with replacement) from the original population to get a measure of how much variability can be expected

def bootstrapping(x, n_samples, func=np.mean):
    y = x.copy()
    N = len(y)
    population = []
    
    for i in range(n_samples):
        population.append(func(np.random.choice(y, N, replace=True)))
        
    return np.array(population)

def histogram(values, n_bins=100):
    xmax = values.max()
    xmin = values.min()
    delta  = (xmax-xmin)/n_bins
    
    counts = np.zeros(n_bins+1, dtype='int')
    
    for value in values:
        val_bin = np.around((value-xmin)/delta).astype('int')
        counts[val_bin] += 1.0
    
    bins = xmin+delta*np.arange(n_bins+1)
    
    return bins, counts/values.shape[0]

x = np.random.normal(0, 2, size=100)

boot = bootstrapping(x, 1_000)

x.mean()

x.std()

bins, counts = histogram(boot)

plt.plot(bins, counts)
plt.vlines(x=boot.mean(), ymin=0, ymax=counts.max(), label='mean', color='g')
plt.vlines(x=boot.mean()+boot.std(), ymin=0, ymax=counts.max(), label='std', linestyles='--', color='r')
plt.vlines(x=boot.mean()-boot.std(), ymin=0, ymax=counts.max(), label='std', linestyles='--', color='r')


