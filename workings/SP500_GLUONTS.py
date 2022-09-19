import pandas as pd
import numpy as np
import pandas_datareader as pdr

df = pdr.get_data_fred("SP500", start="2012-01-01")
df.loc["2017":"2019-05"]

# Downsampling: daily to monthly
# Upsampling: monthly to daily

# Monthly average
df2 = df.resample("M").mean()
df2.head()

df2.pct_change()
np.log(df2)

# %%
# Data Visualisations
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

fig = px.line(df2)
fig.update_layout(yaxis_title="", xaxis_title="Date", showlegend=False)

# %%
# GluonTS
from gluonts.dataset.common import ListDataset
import mxnet as mx
from gluonts.mx.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

ts_train_ld = ListDataset([{'target': df2["SP500"].values, 'start': df2.index[0]}], freq="M")

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=12,
    context_length=24,
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)                          

predictor = estimator.train(ts_train_ld) 

# %%
from itertools import islice
from gluonts.evaluation.backtest import make_evaluation_predictions

import matplotlib.pyplot as plt
def plot_forecasts(tss, forecasts, past_length, num_plots):
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.show()

forecast_it, ts_it = make_evaluation_predictions(ts_train_ld, predictor=predictor)
forecasts = list(forecast_it)
tss = list(ts_it)
plot_forecasts(tss, forecasts, past_length=150, num_plots=3)

# %%
# Produces commonly used error metrics e.g. MSE, MASE, symmetric MAPE, RMSE, (weighted) quantile losses
from gluonts.evaluation import Evaluator

evaluator = Evaluator(quantiles=[0.5], seasonality=2017)

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(ts_train_ld))
agg_metrics
                          
# %%
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

seasonal_predictor_1W = SeasonalNaivePredictor(freq="5min", prediction_length=36, season_length=2016)

forecast_it, ts_it = make_evaluation_predictions(ts_train_ld, predictor=seasonal_predictor_1W)
forecasts = list(forecast_it)
tss = list(ts_it)

agg_metrics_seasonal, item_metrics_seasonal = evaluator(iter(tss), iter(forecasts), num_series=len(ts_train_ld))

df_metrics = pd.DataFrame.join(
    pd.DataFrame.from_dict(agg_metrics, orient='index').rename(columns={0: "DeepAR"}),
    pd.DataFrame.from_dict(agg_metrics_seasonal, orient='index').rename(columns={0: "Seasonal naive"})
)
df_metrics.loc[["MASE", "sMAPE", "RMSE"]]
