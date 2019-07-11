

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data3=pd.read_csv('50_Startups.csv')
data3.columns
x=data3.iloc[:,:4].values
y=data3.iloc[:,4].values



from sklearn.preprocessing import LabelEncoder
lEncoder=LabelEncoder()
x[:,3]=lEncoder.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
ohencoder=OneHotEncoder(categorical_features=[3])

x=ohencoder.fit_transform(x).toarray()
x=x[:,1:]


from sklearn.linear_model import LinearRegression
mRegressor=LinearRegression()

mRegressor.fit(x_train,y_train)

y_pred=mRegressor.predict(x_test)


score=mRegressor.score(x_test,y_test)
from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,y_pred)**(1/2)