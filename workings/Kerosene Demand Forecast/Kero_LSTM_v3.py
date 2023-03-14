#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:45:05 2021

@author: jingwen
"""

import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
# scaling based on quantile, removes median
from pandas.plotting import register_matplotlib_converters

# pip install gdown
# pip install tensorflow

# DEMAND
######### Removed #index_col = Month compared to original
df = pd.read_csv('/Users/jingwen/Desktop/Python_Projects/Japan_SD_Data.csv', sep = ',', parse_dates= ['Month'], index_col='Month')
df.head()
Kero = df[["Month", "Kerosene"]]
cluster = Kero
print(Kero)

# TEMPERATURE
temp = pd.read_html('https://www.data.jma.go.jp/obd/stats/etrn/view/monthly_s3_en.php?block_no=47401&view=3', header=0)
temp = temp[1] #retrieve dataframe from index = 1
df_temp = pd.DataFrame(temp) #convert list to df

# Adjustment
df_temp_loc = df_temp.iloc[64:84] #iloc index 2002 to 2021
df_temp_adj = df_temp_loc.set_index('Year').T #transpose
print(df_temp_adj)

# feature enigneering
df['Month'] = df.index.month
df['day_of_week'] = df.index.dayofweek

# plot
sns.lineplot(x=df.index, y="Kerosene", data=df);
# trough in summer, peak in winter
sns.lineplot(x=df.index.month, y="Kerosene", data=df);
sns.lineplot(x=df.index.dayofweek, y="Kerosene", data=df);

#data processing
#using last 10% of the data for testing
train_size = int(len(Kero) * 0.9)
test_size = len(Kero) - train_size
train, test = Kero.iloc[0:train_size], Kero.iloc[train_size:len(Kero)]
print(len(train), len(test))

f_columns = ['Kerosene']
f_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)
test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)

Kero_transformer = RobustScaler()
Kero_transformer = Kero_transformer.fit(train[['Kerosene']])
train['Kerosene'] = Kero_transformer.transform(train[['Kerosene']])
test['Kerosene'] = Kero_transformer.transform(test[['Kerosene']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train.Kerosene, time_steps)
X_test, y_test = create_dataset(test, test.Kerosene, time_steps)
print(X_train.shape, y_train.shape)

# Predicting Demand with tensorflow
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')