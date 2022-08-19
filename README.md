# Time Series Analysis
## Models
- ARIMA
- SARIMA
- SARIMAX
- TBATS/BATS
- LSTM
- Bi-LSTM

## Key Patterns
Level: average value in the series <br>
Trend: increasing/decreasing value in the series <br>
Seasonality: pattern repeats periodically over time <br>
Cyclical: pattern that increases/decreases but usually non-seasonal 
activity, e.g. business cycles <br>
Noise: random variation in the series

![Image](https://www.bounteous.com/sites/default/files/b_inline_20200914.png)

## Decomposition
Additive seasonal decomposition: level + trend + seasonality + noise (linear-like behaviour) <br>
Multiplicative: level * trend * seasonality * noise (exponential/quadratic/curved) <br>
If seasonality & residual are independent of trend = additive <br>
If seasonality & residual are dependent on trend = multiplicative <br>

```python
import statsmodels.api as sm

def seasonal_decompose(y):
    decomposition = sm.tsa.seasonal_decompose(y, model = 'additive', 
    extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14, 7)
    plt.show()

seasonal_decompose(y)
```

## Check for Stationarity
Stationary: if its statistical properties like mean, variance, and autocorrelation do not change over time
Non-stationary elements like trends and economic cycles
As most time series forecasting models use stationarity - and mathematical transformations related to it - in order to make predictions, we need to **stationarize** to fit the model. <br>

## Augmented Dickey-Fuller (ADF) Test
```python
from statsmodels.tsa.stattools import adfuller

def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

ADF_test(y, 'raw data')
```

## Stationarize data
Detrending, differencing

```python
# Detrending
y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

test_stationarity(y_detrend,'de-trended data')
ADF_test(y_detrend,'de-trended data')
```

```python
# Differencing
y_12lag =  y - y.shift(12)

test_stationarity(y_12lag,'12 lag differenced data')
ADF_test(y_12lag,'12 lag differenced data')
```

```python
# Detrending + Differencing
y_12lag_detrend =  y_detrend - y_detrend.shift(12)

test_stationarity(y_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y_12lag_detrend,'12 lag differenced de-trended data')
```
# SARIMA/SARIMAX with Fourier terms
Two major drawbacks:
(1) One model can only have a single seasonal effect
(2) Seasonal length should not be too long

With SARIMAX, using exogenous variables to model, additional seasonalities with Fourier terms.

# TBATS
Preferable with seasonality that changes over time.
Deal with complex seasonalities (e.g. non-integer, non-nested, large-period seasonality) for long-term forecast

