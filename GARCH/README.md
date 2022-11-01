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

## Difference between ARCH/GARCH

| ARCH                                                         | GARCH                                             |
| -----------                                                  | -----------                                       |
| AR model with conditional heteroskedasticity                 | Model variance with AR(p), generalised ARCH       |
| Does not consider volatility of the previous period          | Considers volatility of the previous period       |
| Does not model change in variance over time                  | Models conditional change in variance over time   |


## **STL Decomposition**
"Seasonal and Trend decomposition using Loess" <br>
Loess used to estimate nonlinear relationships
> For multiplicative decompositions: take logs of data, then back-transforming components. Decompositions between additive and multiplicative can be obtained using a Box-Cox transformation with 0 < λ < 1. A value of λ = 0 gives a multiplicative decomposition while λ = 1 gives an additive decomposition.


| Pros                                             | Cons                                                               |
| -----------                                      | -----------                                                        |
| STL will handle any type of seasonality          | Does not handle trading day/calendar variation automatically       |
| Seasonal component allowed to change over time   | Only for additive decompositions                                   |
| Can be robust to outliers                        |                                                                    |

