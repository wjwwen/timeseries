# Time Series Analysis
## Models
- ARIMA
- SARIMA/SARIMAX
- TBATS/BATS
- LSTM/Bi-LSTM
- GARCH/ARCH
- GLUONTS (Apache MXNet)
- VAR/VECM

**Level**: average value in the series <br>
**Trend**: increasing/decreasing value in the series <br>
**Seasonality**: pattern repeats periodically over time <br>
**Cyclical**: pattern that increases/decreases but usually non-seasonal 
activity, e.g. business cycles <br>
**Noise**: random variation in the series

![Image](https://www.bounteous.com/sites/default/files/b_inline_20200914.png)

**ACF**: Autocorrelation Function (ACF) <br>
**PACF**: Partial Autocorrelation Function (PACF)

## Decomposition
**Additive**: level + trend + seasonality + noise (linear-like behaviour) <br>
**Multiplicative**: level * trend * seasonality * noise (exponential/quadratic/curved) <br>
If seasonality & residual are independent of trend, use additive <br>
If seasonality & residual are dependent on trend, use multiplicative <br>

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
> Stationary: if its statistical properties like mean, variance, and autocorrelation do not change over time <br>
Non-stationary: trends and economic cycles <br><br>
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
Detrending and differencing

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

# Autoregression Models
Time series is modeled as a linear combination of its own lags, i.e. the past values of the series are used to forecast the current/future
- ARIMA/SARIMA/SARIMAX

# SARIMA/SARIMAX with Fourier terms
Two major drawbacks: <br>
(1) One model can only have a single seasonal effect <br>
(2) Seasonal length should not be too long <br>

With SARIMAX, using exogenous variables to model, additional seasonalities with Fourier terms.

# TBATS
Preferable with seasonality that changes over time.
Deal with complex seasonalities (e.g. non-integer, non-nested, large-period seasonality) for long-term forecast

# LSTM/BI-LSTM
Capable of learning long-term dependencies, especially in sequence prediction problems. <br>
No need to check for stationarity/correct it, but stationary data lends support for better neural network learing. <br>
BI-LSTM: backwards/forwards sequence information

# GARCH/ARCH
- Generalised Autoregressive Conditional Heteroskedasticity Model
- Explains considitional variance that changes over time
- Excellent accuracy in short-term forecasting, hard to apply for long-term
- Commonly used to estimate the volatility of returns for stocks/currencies
- GARCH considers volatility of the previous period, ARCH does not
- GARCH assume variance of the error term follows ARMA

![Image](https://cdn.corporatefinanceinstitute.com/assets/heteroskedasticity.png)

# GLUONTS
- AWS Model based on Gluon interface to Apache MXNet
- Apache MXNet: open-source deep learning framework used to train and deploy NN

## DeepAREstimator
- **Estimator**: represents a model that can be trained on a dataset to yield a Predictor 
- **Predictor**: can later be used to make predictions on unseen data
- **Evaluator**: model evaluation (MSE/MASE/symmetric MAPE/RMSE, weighted quantile losses)

Documentations:
- https://aws.amazon.com/blogs/machine-learning/creating-neural-time-series-models-with-gluon-time-series/
- https://aws.amazon.com/blogs/machine-learning/introducing-gluon-an-easy-to-use-programming-interface-for-flexible-deep-learning/

# VAR/VECM
- Vector Auto-Regressive (VAR) - when two/more time series infleunce each other
- Vector Error Correction Model (VECM)
- e.g. Forecast prices using interrelationship beween futures and spot price

|                      | ARIMA               | VAR            |    
| -------------        | -------------       |:-------------: | 
| Variables influence  | Uni-directional     | Bi-directional | 

## VAR Model
**1. Analyse time series characteristics**
<br> Import dataset/data visualise <br><br>
**2. Test for causation amongst the time series**
<br> Granger's Causality test
<br> Cointegration test <br><br>
**3. Split the series into train/test set** <br><br>
**4. Check for stationarity/transform to make stationary, if needed**
<br> ADF test <br><br>
**5. Find optimal order (p)**
<br> AIC, BIC, FPE, HQIC <br><br>
**6. Train the VAR Model of selected order (p)** <br><br>
**7. Check for Serial Correlation of Residuals (errors)** 
<br> Durbin Watson statistic <br><br>
**8. Forecast**
<br> Statsmodels <br><br>
**9. Invert the transformation to get real forecast** <br><br>
**10. Plot forecast v.s. actual** <br><br>
**11. Evaluate forecast**
<br> MAPE, ME, MAE, MPE, RMSE, corr, minmax