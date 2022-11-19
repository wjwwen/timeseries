# Week 1: Linear Regression Models
• Overview – Linear Regression Models <br>
• Overview – Model Building Process <br>
• Types of Data <br>
• Structure Time Series Model <br>
• Statistical Package - EViews

y = f(x1,...,xn) + u
- Separate structures from random noise and variations
- Regression: estimate f; functional relationships
- Infinite number of nonlinear relationship but only 1 linear rs

**Econometrics** is the application of statistical methods to economic data 
in order to give empirical content to economic relationships
> **Unbiasedness**
**Efficiency** 
**Consistency**
**Estimation**

> Gauss Markov Theorem: 
Ordinary Least Squares (OLS) method to estimate beta parameters,
giving the best method among all the models. Provides unbiased, most efficient, consistent model.
<br> Best Linear Unbiased Estimator (BLUE) 

> Simple return v.s. log returns: most likely using the log returns (continuously compounded returns)

## Lecture Slides
Regression Analysis for:
1. Descriptive
2. Predictive (interpretation/possible causal relationship)

| Time Series                                                                                   | Cross Sectional                                             | Panel Data                                                     |
| -----------                                                                                   | -----------                                                 | -----------                                                    |
| GNP/unemployment, government budget deficit, money supply, value of a stock market index      | stock returns on the NYSE, bond credit ratings for UK banks | daily prices of a number of blue chip stocks over the years    |

1. Structured Time Series (STM) - y = Bo + B1X1 + B2X2,... BnXn + Ut
2. Pure Time Series (PTM) - yt = Bo + B1X1 + B2X2,...+ BnXn + Ut
3. Just Y only, (y-1) AR(p) model 
Using Yt historical numbers as input
AR(p)
4. Historical error terms (historical residues to interpret Y)
MA(q)

Steps 3 and 4 = ARMA(p, q)

**Time Series:**
- stock index v.s. macroeconomic fundamentals
- stock price variation when it announced the value of its dividend payment
- effect on a country's currency of an increase in its interest rate

**Cross-Sectional:**
- the relationship between company size and the return to investing in its shares
- country's gdp and probability that the government will default on sovereign debt

**Panel:**
- Both time series and cross sectional
- "Return of the stock is a function of the time series"

> Conditional variation is a function of time T
>> ARCH: model volatility by using historical error terms squared

> What does Jarque-Bera test show?
The Jarque-Bera test is a goodness-of-fit test that determines whether or not sample data have skewness and kurtosis that matches a normal distribution

> Kurtosis is the first moment of the standardised error term

> Maximum Likelihood Method cannot be used here as you need to know the distribution! OLS does not need to know the distribution

> An F-test is any statistical test in which the test statistic has an F-distribution under the null hypothesis. It is most often used when comparing statistical models that have been fitted to a data set, in order to identify the model that best fits the population from which the data were sampled. 
Ho: B1 = B2 = B3 = 0 <br>
If null hypothesis is true, <br> 
y = b0 + u <br>
Model becomes the simplest model.. No linear r/s

> Y noise = random = good model

### Week 2 and 3: Univariate Time Series Modeling and Forecasting
> • Introduction <br>
• Moving Average Processes <br>
• Autoregressive Processes <br>
• ARMA Processes <br>
• Building ARMA models: Box-Jenkins Approach <br>
• Forecasting Using ARMA Models <br>

- Functional relationship between X/Y 
- Distribution of the model 
- How to know the validity of the model? Residues analysis: if purely random, the structure you propose is a good approximation

- Structural Time Series (STM) = Multivariate Time Series
- Pure Time Series (PTM) = Univariate Time Series

### 1. Autoregressive Model
Current value of a variable, y, depends upon only the values that the variable took in previous periods
plus an error term. Stationarity.

An autoregressive model of order p, denoted as AR(p), <br>
yt = μ + φ1 yt−1 + φ2 yt−2 +···+ φp yt−p + ut

### 2. Wold's Decomposition Theorem
Any stationary series can be decomposed into the sum of two unrelated processes, deterministic + purely stochastic, which will be an MA(∞).

### Lecture Notes
- Strictly stationary = If the joint distribution unchanged
- Weakly stationary = covariance stationary 
- Covariance = pair of random variables defined on the same probability space. 
- Autocovariance = pair of values in a discrete-time stochastic process.
- Autocovariances = Covariances of Ys with its own previous values
- Autocorrelation = Correlations X

| Autoregressive                            | Moving Average                         | ARMA           |
| -----------                               | -----------                            | -----------    |                     
| Geometrically decaying acf                | Geometrically decaying pacf            | Geometrically decaying ACF  |
| No. of non-0 points for PACF = AR order   | No. of non-0 points for ACF = MA order | Geometrically decaying PACF |

### Q-Statistics 
- Poor small sample properties (inaccurate/not powerful)
- The Q-statistic is a test statistic output by either the Box-Pierce test/modified version which provides better small sample properties, by the Ljung-Box test. It follows the chi-squared distribution. 

### E-views
- Rejecting null hypothesis is a good news (P-value < 0.05)
- White noise = stationary
- (a) White noise can be used to construct predictable AR or MA process 
- (b) A time series model is adequate (good enough) if residual is white noise

### Lags
- Ts definitely equals to 0 if X variant Q 
- Q = MA models
- lag s
- AR(p): Ts tends to 0, S tends to 0
- MA(q): Ts = 0, S > q
- MA model must be invertible

|                        | AR(p)                                 | MA(q)                               | ARMA(p,q)                          |
| -----------            | -----------                           | -----------                         | -----------                        |         
| ACF = Ts               | Ts tends to 0, S tends to infinity    | Ts = 0, S > q                       | Ts tends to 0
| PACF = Tss             | Tss = 0, S > p                        | Tss tends to 0, S tends to infinity | Tss tends to 0, S tends to infinity

### Week 4: Modelling Long-Run Relationships in Finance
• Stationarity and Unit Root Testing <br>
• Cointegration <br>
• Equilibrium Correction or Error Correction Models <br>
• Testing for Cointegration in Regression <br>

### Week 5 and 6: Modelling Volatility and Correlation
• Models for Volatility <br>
• Autoregressive Conditionally Heteroscedastic (ARCH) Models <br>
• Generalized ARCH (GARCH) Models <br>
• Estimation of ARCH/GARCH Models <br>
• Extensions to the Basic GARCH Model <br>
• GJR and EGRACH Models <br>
• GARCH-in-Mean <br>
• Use of GARCH Models Including Volatility Forecasting <br>

Individual Assignment <br>
ARMA returns <br>
GARCH volatility <br> 
ARMA-GARCH model <br>
Final Exam - proposing a model/opinions