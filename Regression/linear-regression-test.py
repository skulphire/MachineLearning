# regression: y=mx+b
# find out what m and b is

# features and labels
# attributes*

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


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


# print(df.head())

#features = X
#labels = y

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_late = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=5)
clf.fit(X_train, y_train)

with open('LR.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in=open('LR.pickle','rb')
clf = pickle.load((pickle_in))

accuracy = clf.score(X_test,y_test)

#print(forecast_out)
#print(accuracy)

forecast_set =clf.predict(X_late)

#print(forecast_set, accuracy,forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()