# Week 1: Linear Regression Models
>• Overview – Linear Regression Models <br>
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

| Time Series                                                                                   | Cross Sectional                                             | Panel Data                                                     |
| -----------                                                                                   | -----------                                                 | -----------                                                    |
| GNP/unemployment, government budget deficit, money supply, value of a stock market index      | stock returns on the NYSE, bond credit ratings for UK banks | daily prices of a number of blue chip stocks over the years    |

1. **Structured Time Series (STM) aka Multivariate Time Series [y]** = Bo + B1X1 + B2X2,... BnXn + Ut 
2. **Pure Time Series (PTM) aka Univariate Time Series [yt]** = Bo + B1X1 + B2X2,...+ BnXn + Ut
3. **AR(p)** = Just Y only, (y-1) AR(p) model 
Using Yt historical numbers as input
4. **MA(q)** = Historical error terms (i.e. historical residues to interpret Y)

> Steps 3 and 4 = ARMA(p, q)

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

### Conditional variation is a function of time T
ARCH: model volatility by using historical error terms squared

### What does Jarque-Bera test show? <br>
Goodness-of-fit test that determines whether or not sample data have skewness and kurtosis that matches a normal distribution.
Kurtosis is the first moment of the standardised error term

### OLS, not Maximum Likelihood Method 
Maximum Likelihood Method cannot be used here as you need to know the distribution! OLS does not need to know the distribution

### F-distribution under the null hypothesis
It is most often used when comparing statistical models that have been fitted to a data set, in order to identify the model that best fits the population from which the data were sampled. <br>
Ho: B1 = B2 = B3 = 0 <br>
If null hypothesis is true, <br> 
y = b0 + u <br>
Model becomes the simplest model... No linear r/s

### Y noise = random = good model

# Week 2 and 3: Univariate Time Series Modeling and Forecasting
> • Introduction <br>
• Moving Average Processes <br>
• Autoregressive Processes <br>
• ARMA Processes <br>
• Building ARMA models: Box-Jenkins Approach <br>
• Forecasting Using ARMA Models <br>

### Residues analysis to identify validity of the model
Via residues analysis, if purely random, the structure you propose is a good approximation

## 1. Autoregressive Model (AR)
Current value of a variable, Y, depends upon only the values that the variable took in previous periods plus an error term. Requires Stationarity.

> An autoregressive model of order p, denoted as AR(p), <br>
yt = μ + φ1 yt−1 + φ2 yt−2 +···+ φp yt−p + ut

## 2. Wold's Decomposition Theorem (MA)
Any stationary series can be decomposed into the sum of two unrelated processes = **deterministic + purely stochastic** 

## Keywords
- Strictly stationary = If the joint distribution unchanged
- Weakly stationary = covariance stationary 
- Covariance = pair of random variables defined on the same probability space. 
- Autocovariance = pair of values in a discrete-time stochastic process.
- Autocovariances = Covariances of Ys with its own previous values
- Autocorrelation = Correlations X

## Summary of AR, MA, and ARMA
| Autoregressive (AR)                            | Moving Average (MA)                    | ARMA           |
| -----------                                    | -----------                            | -----------    |                     
| Geometrically decaying acf                     | Geometrically decaying pacf            | Geometrically decaying ACF  |
| No. of non-0 points for PACF = AR order        | No. of non-0 points for ACF = MA order | Geometrically decaying PACF |

### Q-Statistics
- Poor small sample properties (inaccurate/not powerful)
- The Q-statistic is a test statistic output by either the Box-Pierce test/modified version which provides better small sample properties, by the Ljung-Box test. It follows the chi-squared distribution. 

### Lags
- Ts definitely equals to 0 if X variant Q 
- Q = MA models
- lag s
- AR(p): Ts tends to 0, S tends to 0
- MA(q): Ts = 0, S > q
- MA model must be invertible

|                          | AR(p)                                 | MA(q)                               | ARMA(p,q)                           |
| -----------              | -----------                           | -----------                         | -----------                         |         
| ACF aka Ts               | Ts tends to 0, S tends to infinity    | Ts = 0, S > q                       | Ts tends to 0                       |
| PACF aka Tss             | Tss = 0, S > p                        | Tss tends to 0, S tends to infinity | Tss tends to 0, S tends to infinity |

## EViews
- Rejecting null hypothesis is a good news (P-value < 0.05)
- White noise = stationary
- (a) White noise can be used to construct predictable AR or MA process 
- (b) A time series model is adequate (good enough) if residual is white noise

### Detecting Serial Correlation
1. With Correlogram diagram, if there is no serial correlation, Autocorrelation and Partial Correlation at all lags **should be near zero** and all (Ljung-Box) **Q-statistics should be insignificant**. A variable that is serially correlated indicates that it **is not random.** 
2. Durbin-Watson (e.g. DW = 0.02768, reject null hypothesis of no serial correlation.)
3. Breusch-Godfrey

# Readings: Chapter 6
- **Stationary** = constant mean, variance, autocovariance structure
- **Autocovariance** = determine how y is related to its previous values
- **s** = lag 
- **c** = constant
- **Autocorrelations (acf)** = autocovariance normalised (by dividing by the variance)
- **White noise** = each observation uncorrelated with all other values in the sequence
- **MA model** = linear combinaton of white noise processes
- **Partial Autocorrelation (pacf)** = direct connections between yt and yt-s for s <= p

> PACF for MA = invertibility (also kinda like stationarity for AR)
<br> ACF for AR = stationarity

## Building ARMA models: the Box-Jenkins approach
1. Identification
2. Estimation
3. Diagnostic Checking

### 1. Information Criteria
1. Function of residual sum of squares (RSS)
2. Some penalty for the loss of degrees of freedom from adding extra parameters

e.g. Adding new variable/lag >>> RSS falls (more than) penalty term increases >>> information criteria value reduces (the lower, the better)

- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (SBIC)
- Hannah-Quinn Criterion (HQIC)

AIC (less strict penalty) > HQIC > SBIC (strict penalty)

### 2. Estimation
### 3. Diagnostic Checking
- Overfitting and Residual Diagnostics
- Examining if residuals are free from autocorrelation

## ARIMA Modelling
- 'I': Integrated
> I stands for the no. of times differencing is needed to make the time series stationary. Applicable for real life as most data are non-stationary and need differencing.
- ARMA(p, q)
- ARIMA(p, d, q)

# Week 4: Modelling Long-Run Relationships in Finance
• Stationarity and Unit Root Testing <br>
• Cointegration <br>
• Equilibrium Correction or Error Correction Models <br>
• Testing for Cointegration in Regression <br>

## Lecture Slides
- Optimal forecasting = minimizing the mean squared 
- Conditional expectations to forecast
- Rolling windows, one step hat to forecast in the v
- Theil’s U-statistic (1966): A U-statistic of one implies that the model under consideration and the benchmark model are equally (in)accurate, while a value of less than one implies that the model is superior to the benchmark, and vice versa for U > 1.
- Time-series Econometrics: Cointegration and Autoregressive Conditional Heteroskedasticity
- Spurious Regressions in Econometrics (CWJ Granger and P Newbold) - 1973
- While the Durbin-Watson test is aimed at an autocorrelation of 1st order, the Breusch-Godfrey-test can also uncover autocorrelation of higher orders. In section 6.4 we then present various approaches to the estimation of multiple regression model with autocorrelated disturbance variables.

- Yt = U1 + Yt-1 + Ut
- Xt = u2 + Xt-1 + Ut
- Both random walks are independent and non-stationary

# Readings: Chapter 8
- Stationary = constant mean, variance, autocovariance
- Non-stationary data can lead to spurious regressions
- Non-invertible MA = cannot be expressed in an autoregressive process

Check if Y contains unit root:
- I(0) = stationary
- I(1) = 1 unit root, requires 1 differencing to induce stationarity
- I(2) = 2 unit roots, require 2 differencing to induce stationarity
- Majority just 1 unit root
- 2 unit roots e.g.: nominal consumer prices and nominal wages

### Dickey-Fuller Test for Unit Root:
- Change in yt = ψyt−1 + ut
- test of φ = 1 is equivalent to a test of ψ = 0 (since φ − 1 = ψ).
- H0: series contains unit root
- H1: series stationary

### Philips-Perron (PP) tests:
- More comprehensive theory of unit root non-stationarity
- Considers unit roots in presence of known **structural breaks**
- Structural breaks e.g.: changes in monetary policy/removal of exchange rate controls
- "... short-term interest rates are best viewed as **unit root processes that have a structural break in their level around the time of Black Wednesday (1992) when the UK dropped out of the European Exchange Rate Mechanism. The longer term rates, on the other hand, are I(1) processes with no breaks"

### Summary of ADF, PP, KPSS
| ADF/PP          | KPSS                    | 
| -----------     | -----------             |                 
| H0: yt ∼ I(1)   | H0: yt ∼ I(0)           | 
| H1: yt ∼ I(0)   | H1: yt ∼ I(1)           | 

There are four possible outcomes: <br>
(1) Reject H0 and Do not reject H0 <br>
(2) Do not reject H0 and Reject H0 <br>
(3) Reject H0 and Reject H0 <br>
(4) Do not reject H0 and Do not reject H0 <br>

For robust conclusions, results should fall under (1) or (2), <br>
when both tests conclude that series = stationary. <br>

<i> ADF = left-tailed test, KPSS = right-tailed test </i>

### Seasonal Unit Roots
- Seasonal differencing to induce stationarity
- I(d, D)
- d = no. of differencing
- D = no. of seasonal differencing
- Not widely adopted, as data better characterised with dummy variables

### EViews Unit Root
e.g. t-Statistic
- Augmented Dickey-Fuller test statistic = -0.47
- Test critical values: 
- 1% level - -3.45
- 5% level - -2.87
- 10% level - -2.57
As the test statistic < critical values, the null hypothesis of a unit root cannot be rejected

### Cointegration
> I(1) variables will be I(0) = stationary
- Error correction model = equilibrium correction model

# Week 5 and 6: Modelling Volatility and Correlation
• Models for Volatility <br>
• Autoregressive Conditionally Heteroscedastic (ARCH) Models <br>
• Generalized ARCH (GARCH) Models <br>
• Estimation of ARCH/GARCH Models <br>
• Extensions to the Basic GARCH Model <br>
• GJR and EGRACH Models <br>
• GARCH-in-Mean <br>
• Use of GARCH Models Including Volatility Forecasting <br>

# Misc
Individual Assignment <br>
ARMA returns <br>
GARCH volatility <br> 
ARMA-GARCH model <br>
Final Exam - proposing a model/opinions