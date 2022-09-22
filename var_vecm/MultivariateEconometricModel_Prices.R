# Multivariate Econometric Model Configuration
# VAR/VECM
# Volatility, granger causality, error correction

library(tidyverse)
library(alfred)
library(mice) # Multivariate Imputation by Chained Equation
library(VIM)
library(tseries)
library(vars)
library(tsDyn) # Nonlinear Time Series Models with Regime Switching

wti = as_tibble(get_fred_series("DCOILWTICO", "WTI", observation_start = "2000-01-02"))
brent = as_tibble(get_fred_series("DCOILBRENTEU", "Brent", observation_start = "2000-01-02"))
hh = as_tibble(get_fred_series("MHHNGSP", 'HenryHub', observation_start = "2000-01-02"))
oil_prices = wti %>% left_join(brent, by="date")
prices = hh %>% left_join(oil_prices, by="date")
head(prices)

# understanding missing value patterns
md.pattern(prices)

# visualize missing values
marginplot(prices[, c("WTI", "Brent")], col = mdc(1:2), cex.numbers = 1.2, pch = 19)

# input missing values
imputes = mice(prices, m=5, maxit=40)
# methods for used for imputing
imputes

# imputed dataset
data = complete(imputes, 5)
head(data)

# goodness of fit
xyplot(imputes, WTI ~ Brent | .imp, pch = 20, cex = 1.4)

# density plot
densityplot(imputes)

# -----------------------------------------------------------------------------
# Time Series and Line Plots
# converting to time -series
henryhub = ts(data$HenryHub, start = c(2000,02,01), frequency = 252)
wti = ts(data$WTI, start = c(2000,02,01), frequency = 252)
brent = ts(data$Brent, start = c(2000,02,01), frequency = 252)
# time series line plot
par(mfrow=c(3,1), mar=c(0,3.5,0,3), oma=c(3.5,0,2,0), mgp=c(2,.6,0), cex.lab=1.1, tcl=-.3, las=1)
plot(henryhub, ylab=expression('HenryHub price'), xaxt="no", type='n')
grid(lty=1, col=gray(.9))
lines(henryhub, col=rgb(0,0,.9))

plot(wti, ylab=expression('WTI crude price'), xaxt="no", yaxt='no', type='n')
grid(lty=1, col=gray(.9))
lines(wti, col=rgb(.9,0,.9)) 
axis(4) 
plot(brent, ylab=expression('Brent Crude price'))
grid(lty=1, col=gray(.9))
lines(brent, col=rgb(0,.7,0))
title(xlab="Date", outer=TRUE)
df = cbind(henryhub, wti, brent)

# -----------------------------------------------------------------------------
# ADF unit root test
# Calculated statistics > tabulated values, null hypotheses can be rejected
# ADF test (H0: series has unit root)
# Philiips-Perron (PP) test (H0: series has unit root)
# KPSS test (H0: series has no unit root)
# Zivot and Andrew test (H0: series has unit root)

adf.test(log(data[, "HenryHub"]))
adf.test(log(data[, "WTI"]))
adf.test(log(data[, "Brent"]))

pp.test(log(data[, "HenryHub"]), type = "Z(t_alpha)")
pp.test(log(data[, "WTI"]), type = "Z(t_alpha)")
pp.test(log(data[, "Brent"]), type = "Z(t_alpha)")

kpss.test(log(data[, "HenryHub"]))
kpss.test(log(data[, "WTI"]))
kpss.test(log(data[, "Brent"]))

# -----------------------------------------------------------------------------
# Optimal lags = 5 according to AIC and FPE
# For all five lag orders, a VAR including a constant + trend 
# as deterministic regressors
VARselect(log(df), lag.max=10, type="const")

var_mod <- VAR(df, p = 5, type="both")
summary(var_mod)
plot(var_mod, names="HenryHub")

# -----------------------------------------------------------------------------
# Portmanteau goodness-of-fit test: test adequacy of fitted model 
# checking if residuals are white noise

residuals = serial.test(var_mod, lags.pt=5, type="PT.asymptotic")
residuals$serial

# null hypothesis (no autocorrelation) is rejected as p-value < 0.05

# -----------------------------------------------------------------------------
# Normality test (i.e. following a normal distribution or not)
# Null hypothesis of Jarque-Bera test: skewness = 0 + excess kurtosis = 0
# Reject null hypothesis as p-value < 0.05
# Thus, conclude that residuals does not follow a normal distribution
norm <- normality.test(var_mod)
norm$jb.mul

# -----------------------------------------------------------------------------
# Conditional volatility model
# Since p-value < 0.05, model suffers from heteroscedasticity
arch <- arch.test(var_mod, lags.multi=5, multivariate.only=TRUE)
arch$arch.mul

# -----------------------------------------------------------------------------
# Testing structural breaks
# Assessing stability of coefficients
# Lines do not exceed red lines = stable
plot(stability(var_mod, type = "Rec-CUSUM"))

# -----------------------------------------------------------------------------
# Granger Causality

causality(var_mod, cause = 'wti')
causality(var_mod, cause = 'brent')
causality(var_mod, cause = 'henryhub')

# -----------------------------------------------------------------------------
# Impulse Response
plot(irf(var_mod, impulse="wti", response=c("henryhub"), n_ahead=15, boot=TRUE))

# -----------------------------------------------------------------------------
# Forecast error variance decomposition
# Examining impact of variables on one another
plot(fevd(var_mod, n.ahead = 15))

# Almost 100% of the variance in HenryHub is caused by HenryHub itself
# 80% of variance of WTI/Brent caused by themselves and others

# -----------------------------------------------------------------------------
# Forecast VAR
forecast <- predict(var_mod, n.ahead = 15, ci = 0.95)
fanchart(forecast, names = "henryhub", main = "HenryHub price forecast", xlab = "Horizon", ylab = "Price")
forecast

# -----------------------------------------------------------------------------
# Cointegration test
# If variables are cointegrated, work with error correction model (ECM)

# Eigen test
coin = ca.jo(df, type = "eigen", ecdet = "none", K = 3, spec = "transitory")
summary(coin)

# Trace test
coin = ca.jo(df, type = "trace", ecdet = "none", K = 3, spec = "transitory")
# endet = 'none' means there is a linear trend in data
summary(coin)

s = 1*henryhub - 0.7788110 * wti + 0.6723858*brent + 0.4075001
plot(s, type ='l')

# -----------------------------------------------------------------------------
# ECM model
model_vecm = VECM(df, lag=3, r=1, estim = 'ML')
summary(model_vecm)

# VAR representation of VECM
VARrep(model_vecm)

# Forecast VECM
forecast=predict(model_vecm, n.ahead=15)
forecast