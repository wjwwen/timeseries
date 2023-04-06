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
# Predict S&P500 for the next 12 months
'''
Libraries:
1. statsmodels
2. prophet (by Facebook)
3. GluonTS (by AWS)
4. R Forecast
5. Regression
'''

# %%
# GluonTS
from gluonts.dataset.common import ListDataset
import mxnet as mx
from gluonts.mx.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

ts_train_ld = ListDataset([{'target': df2["SP500"].values, 'start': df2.index[0]}], freq='M')

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimesnions=[10],
    prediction_length=12,
    context_length=24,
    freq="M",
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)                          

predictor = estimator.train(ts_train_ld) 

# %%
ts_train = df2["SP500"].iloc[:-12]
ts_test = df2["SP500"].iloc[-12:]

ts_train

from gta.forecast import GluonForecast
model_sff = GluonForecast(
    method="SimpleFeedForward",
    horizon=12,
    trainer=dict(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)

model_sff = model_sff.fit(ts_train)
pred_sff = model_sff.predict()
pred_sff

                           