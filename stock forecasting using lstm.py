#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd 


# In[56]:


# import yfinance as yf

# # Define the stock ticker (e.g., Apple - AAPL)
# ticker = "AAPL"

# # Download data (adjust the start and end dates as needed)
# df = yf.download(ticker, start="2000-01-01", end="2025-02-11")

# # Display the first few rows
# print(df.head())


# In[58]:


df=pd.read_csv("C:/Users/hp/Downloads/stock_data.xls")
#stock data from 2020 to 14 feb 2025(real data from yfinance)


# In[31]:


#df.to_excel("C:/Users/hp/Downloads/stock_data.xls.xlsx", index=False)  # Saves as Excel without the index column


# In[61]:


#preprocessing the data 
df.rename(columns={"Price": "Date"}, inplace=True)
df = df.iloc[1:].reset_index(drop=True)


# In[64]:


df.tail()


# In[86]:


#filter only one price it may be closing, opening , low , high 
df1=df.reset_index()['Close']


# In[88]:


print(df1.dtypes)


# In[91]:


df1.shape


# In[97]:


df1 = pd.to_numeric(df1, errors="coerce")  # Convert to float64.


# In[98]:


df1


# In[101]:


import matplotlib.pyplot as plt

plt.plot(df1)
plt.xlabel("no. of Days")  # Label for X-axis
plt.ylabel("Price of Stocks")  # Label for Y-axis
plt.title("Price Trend")  # Title of the graph
plt.show()


# In[ ]:


##LSTM are sensitive to the scale of the data . So we apply MinMax scalar


# In[102]:


import numpy as np


# In[103]:


#Subodh Kumar
from sklearn.preprocessing import MinMaxScaler
#transforming values to 0:1 because in lstm dta should be normalize if not, optimal accuracy will not be achieved
scaler=MinMaxScaler(feature_range=(0,1))
#Converts df1 (a Pandas Series) into a NumPy array.
#.reshape(-1,1) ensures the data is in 2D format (required by MinMaxScaler).
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[104]:


df1


# In[105]:


#splitting dataset into train and test split 
#70% of the data in training and 30 of the data in test.
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
#subodh kumar
#train_data contains from 0 to training_size and test_data contains training_size to len(df1)
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[106]:


training_size,test_size


# In[107]:


train_data


# In[108]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[109]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[112]:


print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(ytest.shape)


# In[111]:


# reshape input to be [samples, time steps, features] which is required for LSTM
#Why is This Important?
#->LSTM expects a 3D input format to recognize temporal dependencies.
#->This ensures compatibility with the Keras LSTM layer.
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[114]:


get_ipython().system('pip install tensorflow')


# In[115]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[116]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[117]:


model.summary()


# In[118]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[120]:


import tensorflow as tf


# In[121]:


tf.__version__


# In[122]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[123]:


##Transformback to original form
#Since the data was scaled using MinMaxScaler(feature_range=(0,1)), the predictions are in a 
#normalized form (between 0 and 1). To get the actual stock prices, we need to inverse transform them back.
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[124]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[125]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[126]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[127]:


len(test_data)


# In[164]:


x_input=test_data[287:].reshape(1,-1)
x_input.shape


# In[165]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[166]:


temp_input


# In[167]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[168]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[169]:


len(df1)


# In[170]:


plt.plot(day_new,scaler.inverse_transform(df1[1188:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[171]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[155]:


#subodh Kumar
df3=scaler.inverse_transform(df3).tolist()


# In[ ]:





# In[ ]:




