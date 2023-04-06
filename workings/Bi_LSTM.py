import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

df = pd.read_csv('prices.csv') 
df["WTI"]=pd.to_numeric(df.WTI,errors='coerce') 
df["Gold"]=pd.to_numeric(df["GOLD"],errors='coerce') 
df = df.dropna() 

# %%
# DATA PREPROCESSING
# train 80%, test 20%
price_data = df.loc[:, ["WTI", "Gold"]].values
price_data = price_data.reshape((-1,1)) 

split_percent = 0.80
split = int(split_percent*len(price_data))

price_train = price_data[:split]
price_test = price_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

print(len(price_train))
print(len(price_test))

# %%
trainData = price_train

sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
trainData.shape

X_train = []
y_train = []

for i in range (60,7758): #60 : timestep // 1149 : length of the data
    X_train.append(trainData[i-60:i, 0]) 
    y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1)) #adding the batch_size axis
X_train.shape

# %% 
from keras.layers import Bidirectional
model = Sequential()

model.add(Bidirectional(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1))))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=100, return_sequences = True)))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=100, return_sequences = True)))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=100, return_sequences = False)))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer='adam',loss="mean_squared_error")

hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=2)

# %%

testData = pd.read_csv('GOOG1.csv') #importing the test data
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce') #turning the close column to numerical type
testData = testData.dropna() #droping the NA values
testData = testData.iloc[:,4:5] #selecting the closing prices for testing
y_test = testData.iloc[60:,0:].values #selecting the labels 
#input array for the model
inputClosing = testData.iloc[:,0:].values 
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestep = 60
for i in range(timestep,length): #doing the same preivous preprocessing 
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_test.shape

y_pred = model.predict(X_test) #predicting the new values

predicted_price = sc.inverse_transform(y_pred) #inversing the scaling transformation for ploting
plt.plot(y_test, color = 'black', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'red', label = 'Predicted Stock Price')
plt.title('GOOGLE Stock Price Prediction - BI-LSTM Model')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()