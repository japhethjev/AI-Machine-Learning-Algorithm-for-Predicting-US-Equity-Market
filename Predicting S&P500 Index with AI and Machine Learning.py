#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT THE REQUIRED PYTHON LIBRARIES

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import yfinance as yf


# In[2]:


pip install -U pandas


# In[3]:


#IMPORT STOCK PRICE DATA FROM YAHOO FINANCE

SP500= yf.download("SPY", start="2010-01-01", end= "2022-07-28")
print(SP500)


# In[4]:


#CLEAN AND PROCESS THE DATA BY REMOVING UNWANTED VARIABLES

x = SP500.drop(columns =['Close','Adj Close'])
y= SP500.Close


# In[5]:


print(x)


# #VISUALIZE THE STOCK PRICE DATA TO SEE HOW THE STOCK IS TRENDING

# In[6]:


plt.figure(figsize=(16,8))
plt.title('S&P500 Index')
plt.xlabel('Date')
plt.ylabel('Close Price_USD')
plt.plot(SP500['Close'])
plt.show()


# In[7]:


#SPLIT THE DATA INTO TRAINING DATA AND TESTING DATA IN THE RATIO 70:30

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=101)


# In[8]:


#BUILD THE REGRESSION MODEL AND TRAIN THE MODEL

model = LinearRegression()
model.fit(x_train, y_train)


# In[9]:


#CHECK THE MODEL COEFFICIENTS

print(model.coef_)
print(model.intercept_)


# In[10]:


input_data = np.array([[112.370003,113.389999,111.510002,118944600]])


# In[11]:


predicted_y = model.predict(x_test)


# In[12]:


#PREDICT THE CLOSE PRICE OF THE STOCK AND COMPARE THE PREDICTED CLOSE PRICE WITH ACTUAL CLOSE PRICE

print(predicted_y)


# In[13]:


dframe_predicted = pd.DataFrame({'Actual':y_test,'predicted':predicted_y})


# In[14]:


dframe_predicted.head(25)


# In[15]:


import math


# In[16]:


#EVALUATE THE MODEL TO CONFIRM ITS LEVEL OF ACCURACY USING MODEL EVALUATION METRICS

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicted_y))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,predicted_y))
print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicted_y)))   


# In[17]:


#VISUALIZE THE PREDICTED STOCK CLOSE PRICE VS ACTUAL STOCK PRICE ON A BAR GRAPH

graph=dframe_predicted.head(20)
graph.plot(kind='bar')


# In[ ]:




