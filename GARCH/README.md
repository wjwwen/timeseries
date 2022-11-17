# ARCH/GARCH
## **ARCH: Autoregressive Conditional Heteroskedasticity**
> Use cases: stock prices, oil prices, bond prices, inflation rates,  GDP, unemployment rates

| Autoregressive   | Conditional                   | Heteroskedasticity |
| -----------      | -----------                   | -----------        |
| Correlation      | Variance based on past erorrs | Varying variance   |

## **GARCH: Generalised Autoregressive Conditional Heteroskedasticity Model**
> GARCH is the "ARMA Equivalent" of ARCH
- Incorporates ***moving average** component* 
- Better fit for data exhibiting heteroskedasticity and volatility clustering
> Use cases: electricity forecasting, time series with price spike; long-term memory but gives recent evens more weight

GARCH(p,q)
- p: no. of autoregressive lags imposed on the equation
- q: no. of moving average lags specified

The univariate GARCH models considered are used to model the conditional variance. However, these models do not consider how the conditional covariance varies over time. Therefore, multivariate GARCH models are used to model the conditional covariance matrix in this study.

## Difference between ARCH/GARCH

| ARCH                                                         | GARCH                                             |
| -----------                                                  | -----------                                       |
| AR model with conditional heteroskedasticity                 | Model variance with AR(p), generalised ARCH       |
| Does not consider volatility of the previous period          | Considers volatility of the previous period       |
| Does not model change in variance over time                  | Models conditional change in variance over time   |

# GJR-Garch
The Glosten-Jagannathan-Runkle GARCH(GJR-GARCH) model assumes a specific parametric form for this conditional heteroskedasticity. 

Besides leptokurtic returns, the GJR-GARCH model, like theGARCH model, captures other stylized facts in financial time series, like volatility clustering. 

There is a stylized fact that the GJR-GARCH model captures that is not contemplated by the GARCH model, which is the empirically observed fact that negative shocks at timet−1 have a stronger impact in the variance at time t than positive shocks. This asymmetry used to be called leverage effect because the increase in risk was believed to come from the increased leverage induced by a negative shock, but nowadays we know that this channel is just too small. Notice that the effective coefficient associated with a negative shock isα+γ. In financial time series, we generally find that γ is statistically significant.

The best model (p and q) can be chosen, for instance, by Bayesian Information Criterion (BIC), also known as Schwarz Information Criterion (SIC), or by Akaike Information Criterion (AIC). The former tends to be more parsimonious than the latter. V-Lab usesp=1 andq=1 though, because this is usually the option that best fits financial time series.

## **STL Decomposition**
"Seasonal and Trend decomposition using Loess" <br>
Loess used to estimate nonlinear relationships
> For multiplicative decompositions: take logs of data, then back-transforming components. Decompositions between additive and multiplicative can be obtained using a Box-Cox transformation with 0 < λ < 1. A value of λ = 0 gives a multiplicative decomposition while λ = 1 gives an additive decomposition.


| Pros                                             | Cons                                                               |
| -----------                                      | -----------                                                        |
| STL will handle any type of seasonality          | Does not handle trading day/calendar variation automatically       |
| Seasonal component allowed to change over time   | Only for additive decompositions                                   |
| Can be robust to outliers                        |                                                                    |

