library (quantmod)
library(dplyr)
library(tidyverse)
library(tseries)
library(rugarch)
library(xts)
library(PerformanceAnalytics)

df = getSymbols("TSLA", from="2010-01-01", to="2020-12-31")
chartSeries(TSLA)
chartSeries(TSLA["2020-12"])

return = CalculateReturns(TSLA$TSLA.Adjusted)
# TSLA.Open
# TSLA.High
# TSLA.Low
# TSLA.Close
# TSLA.Volume
# TSLA.Adjusted

return = return[-c(1),] # remove first row without value
chart_Series(return) # time series of returns

# histogram to check normal distribution
# result: more skewed than normal distribution
chart.Histogram(return, method = c('add.density', 'add.normal'), colorset = c('blue','red','black'))
legend("topright", legend = c("return", "kernel", "normal dist"), fill = c('blue', 'red', 'black'))

# Monthly volatility
# Stochastic model
# Stochastic: having a random probability distribution/pattern that may be analysed 
# statistically but may not be predicted precisely
sd(return)
sqrt(252)*sd(return["2020"])
chart.RollingPerformance(R=return["2010::2020"], width=22, FUN="sd.annualized", scale=252, main="TSLA's Monthly Volatility")

# ------------------------------------------------------------------------------------------------------------------------------
# ARMA specification of variance is (0,0)
# Variance models: sGARCH, fGARCH, eGARCH, gjrGARCH, apGARCH, iGARCH, csGARCH
mod_specify = ugarchspec(mean.model = list(armaOrder=c(0,0)), variance.model=list(model="sGARCH", garchOrder=c(1,1)), distribution.model="norm")
mod_fitting = ugarchfit(data=return, spec = mod_specify, out.sample=20)

mod_fitting
plot(mod_fitting, which='all')

# Evaluate mod_fitting:
# 1. Constant parameter Omega1 is insignificant

# 2. Lower values of Information Criteria (e.g. Akaike), the better the model in fitting.

# 3. Ljung-Box test 
# - Null hypothesis = no serial correlation of error terms
# - As p-value > 0.05, do not reject null hypothesis.
# Thus, there is no serial correlation of the error term.

# 4. Adjusted Pearson Goodness-of-Fit 
# - Null hypothesis = conditional error term follows normal distribution
# - As p-value < 0.05, reject null hypothesis.
# Thus, does not follow normal distribution.


# ------------------------------------------------------------------------------------------------------------------------------
# GARCH with skewed student distribution (distribution.model = sstd)
# In real data, residuals do not usually fit normal distribution
# Skewed student distribution is a good fit for error term
# residuals follow normal distribution in QQ plot
mod_specify = ugarchspec(mean.model = list(armaOrder=c(0,0)), variance.model=list(model="sGARCH", garchOrder=c(1,1)), distribution.model="sstd")
mod_fitting = ugarchfit(data=return, spec = mod_specify, out.sample=20)

mod_fitting
plot(mod_fitting, which='all')

# ------------------------------------------------------------------------------------------------------------------------------
# GJR-GARCH(1,1)
mod_specify = ugarchspec(mean.model = list(armaOrder=c(0,0)), variance.model=list(model="gjrGARCH", garchOrder=c(1,1)), distribution.model="sstd")
mod_fitting = ugarchfit(data=return, spec = mod_specify, out.sample=20)

mod_fitting
plot(mod_fitting, which='all')

# ------------------------------------------------------------------------------------------------------------------------------
# OPTIMAL MODEL
# GJR-GARCH(0,1)
# All optimal parameters statistically significant (<0.05)
# Information Criteria all lowest compared to models such as sGARCH
mod_specify = ugarchspec(mean.model = list(armaOrder=c(0,0)), variance.model=list(model="gjrGARCH", garchOrder=c(1,0)), distribution.model="sstd")
mod_fitting = ugarchfit(data=return, spec = mod_specify, out.sample=20)

mod_fitting
plot(mod_fitting, which='all')

forecast = ugarchforecast(fitORspec = mod_fitting, n.ahead=20)
plot(fitted(forecast))
plot(sigma(forecast))

# plot(sigma(forecast))
# run the forecast of the volatility for the next 20 days, 
# expect the volatility of TESLA to potentially increase in the next 5 days 
# and remain at the same level for the remaining days as reflected in the graph.
