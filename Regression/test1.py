# regression: y=mx+b
# find out what m and b is

# features and labels
# attributes*

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression


quandl.ApiConfig.api_key = "3DCuBTNWeAPEK6RotBW-"
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']] #features

#print(df.head())

#label = price

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace = True) # makes outliers for missing data

forecast_out = int(math.ceil(0.01*len(df))) #predict out 10% of df

# label
df['label'] = df[forecast_col].shift(-forecast_out) # adjusting feature negativaly to predict 10 days later

df.dropna(inplace=True)
# print(df.head())

#features = X
#labels = y

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)

print(forecast_out)
print(accuracy)


