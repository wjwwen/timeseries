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
- **MA model** = linear combination of white noise processes
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

## Inverted AR Roots/MA Roots
- The inverses of the AR and MA roots can be used to check whether the process implied by the model is stationary and inverstible.
- To be stationary and invertible, respectively: inverted roots in each case < 1 in absolute value. 

# Week 4: Modelling Long-Run Relationships in Finance
>• Stationarity and Unit Root Testing <br>
• Cointegration <br>
• Equilibrium Correction or Error Correction Models <br>
• Testing for Cointegration in Regression <br>

## Lecture Slides
• Optimal forecasting = minimizing the mean squared <br>
• Conditional expectations to forecast <br>
• Rolling windows, one step hat to forecast in the v <br>
• Theil’s U-statistic (1966): A U-statistic of one implies that the model under consideration and the benchmark model are equally (in)accurate, while a value of less than one implies that the model is superior to the benchmark, and vice versa for U > 1. <br>
•  Time-series Econometrics: Cointegration and Autoregressive Conditional Heteroskedasticity <br>
•  Spurious Regressions in Econometrics (CWJ Granger and P Newbold) - 1973 <br>
•  While the Durbin-Watson test is aimed at an autocorrelation of 1st order, the Breusch-Godfrey-test can also uncover autocorrelation of higher orders. In section 6.4 we then present various approaches to the estimation of multiple regression model with autocorrelated disturbance variables. <br>

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

Advantage of PP test:
- robust to general forms of heteroskedasticity in the error term ut.
- Another advantage is that the user does not have to specify a lag length for the test regression.

### Summary of ADF, PP, KPSS
| ADF/PP unit root tests          | KPSS stationarity test  | 
| -----------                     | -----------             |                 
| H0: yt ∼ I(1)                   | H0: yt ∼ I(0)           | 
| H1: yt ∼ I(0)                   | H1: yt ∼ I(1)           |  

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
As the test statistic > critical values, the null hypothesis of a unit root cannot be rejected

### Cointegration
> I(1) variables will be I(0) = stationary
- Error correction model = equilibrium correction model

# Week 5 and 6: Modelling Volatility and Correlation
>• Models for Volatility <br>
• Autoregressive Conditionally Heteroscedastic (ARCH) Models <br>
• Generalized ARCH (GARCH) Models <br>
• Estimation of ARCH/GARCH Models <br>
• Extensions to the Basic GARCH Model <br>
• GJR and EGRACH Models <br>
• GARCH-in-Mean <br>
• Use of GARCH Models Including Volatility Forecasting <br>

## Engle-Granger methodology 
The first step generates the residuals and the second step employs generated residuals to estimate a regression of first- differenced residuals on lagged residuals. Hence, any possible error from the first step will be carried into second step.

1. All variables are I(1)
2. Use the step 1 residuals as one variable in the error correction model 

# Readings: Chapter 9
## ARCH
ARCH then GARCH
ARCH provides a framework for time series models of volatility.
- First compute the Engle test for Arch effects to make sure that this class of models is appropriate for the data
1. Quick > Estimate Equation > rgbp c ar(1) ma(1) 
2. View > Residual Diagnostics > Heteroskedasticity Tests
"... Both the F-version and the LM-statistic are very significant, suggesting that the presence of ARCH"

Limitations of ARCH:
1. Value of q (no. of lags) of squared residual can be decided with the use of likelihood ratio test (but no clear best approach)
2. Value of q (no. of lags) of squared error might be very large, resulting in large conditional variance model that is not parsimonious
Engle (1982) circumvented this problem by specifying an arbitrary linearly declining lag length on ARCH(4)

- A test for the presence of **ARCH in the residuals** is calculated by regressing the squared residuals on a constant and p lags, where p is set by the user

## GARCH
GARCH-type models can be used to forecast volatility. 
One primary usage of GARCH-type models is in forecasting volatility 
e.g. pricing of financial options where volatility is an input to the pricing model

- GARCH = ARMA model for the conditional variance
- GARCH is more parsimonious, avoid overfitting, less likely to breach non-negativity constraints
- GARCH(1,1) model will be sufficient to capture the volatility clustering in the data, and rarely is any higher order model estimated.
- GARCH (1,1) = ARCH(infinity)
- IGARCH = Integrated GARCH, 'unit root in variance'
- OLS cannot be used for GARCH; OLS minimises the RSS (residue sum of square); RSS depends only on parameters in the conditional mean equation, and not the conditional variance
- Use maximum likelihood - finding the most likely values of parameter values given the actual data. Form log-likelihood function (LLF) 

### Limitations of GARCH
1. Enforce a symmetric response of volatility to positive/negative shocks
(A negative shock to financial time series is likely to cause volatility to rise **more than** a positive shock of the same magnitude)

## Alternatives/Asymmetric formulations:
### 1. (Threshold) T-GARCH (GJR) Model - overcome leverage effects <br>
> Impact of bad news > impact of good news <br>
Alpha + Gamma = Impact of bad news <br>
Alpha = Impact of good news <br>

### 2. EGARCH Model - overcome negativity constraints <br>
EGARCH also covers asymmetric effect/leverage effects <br>
EGARCH is better than TGARCH <br>
- No need to artificially impose non-negativity constraints on the model parameters <br>
> Impact of bad news > impact of good news <br>
Alpha + Gamma = Impact of bad news <br>
Alpha - Gamma = Impact of good news <br>

### 3. GARCH-in-mean
- Incorporate volatility into the mean equation so that we link return to volatility
- Higher risk, higher return

- The return partly determined by its risks
- Conditional variance of asset returns enters into the conditional mean equation

Tests for asymmetries in volatility
1. Engle and Ng - determine whether an asymmetric model is required for a given series, or symmetric GARCH is deemed adequte

## EViews
- The default is to estimate with 1 ARCH and 1 GARCH (i.e. one lag of the squared errors and one lag of the conditional variance, respectively)
- GARCH(1,1) dynamic v.s. static forecasts
- Static forecasts are rolling one-step ahead forecasts for the conditional variance, showing more volatility than dynamic forecasts

# Recap
*** Always propose structural time series model first <br>
*** Either structural/pure, need to check if residues is white noise <br>
*** White noise is distribution free <br>

1. Check Stationarity: Unit Root Test <br>
If I(1) i.e. not stationary, use differencing to make the model stationary <br>
If I(0) all good, no need to use differencing
3. Residues Analysis: Check if proposed model is a good approximation - if residues is white noise
4. Check cointegration: Engle-Granger cointegration test <br>
Part 1 > All variables are I(1) <br>
Part 2 > Use the step 1 residuals as one variable in the error correction model <br>
5. Check variance of the residuals <br>
If variance is unchanged, just use simple time series <br>
If variance is different, use ARCH/GARCH

6. Test for 'ARCH Effects'
- If the squared residuals/errors of your time series model exhibit autocorrelation, then ARCH effects are present.
- If ARCH effects are significant, cannot assume white noise are IID; in fact it is just uncorrelated... Need to use another model
- A time series exhibiting conditional heteroscedasticity—or autocorrelation in the squared series—is said to have autoregressive conditional heteroscedastic (ARCH) effects. Engle's ARCH test is a Lagrange multiplier test to assess the significance of ARCH effects
- If you do not reject null hypothesis >>> use the basic model
- IF you reject null hypothesis, at least one of alpha (i) not equal to 0 >>> use ARCH model

# Miscellaneous
### Multivariate GARCH formulations
- VECH/Diagonal VECH/BEKK models

### Model estimation for multivariate GARCH
- Under the assumption of conditional normality, the parameters of multivariate GARCH models of any of the above specifications can be estimated by maximising the log-likelihood function

Questions:
1. When to use ADF/PP test?
Adf test is used when the errors are homoscedastic and PP test is preferred for heteroscedastic errors.

2. Why QLE (Quasi Likelihood) instead of MLE (Maximum Likelihood) for GARCH? <br>
Using QLE is better.
You can use the typical MLE when you know that your zq do indeed follow a normal distribution. Otherwise, when you can see, for example via appropriate tests, that your zq are not normally distributed, you should use QML (as long as the innovations are iid with zero mean and unit variance).
> See https://stats.stackexchange.com/questions/416964/qml-vs-mle-for-gjr-garch-models