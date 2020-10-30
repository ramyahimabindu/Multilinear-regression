# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:07:06 2020

@author: 91950
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv ('C:/Users/91950/Downloads/startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""## Encoding categorical data"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Avoiding the dummy variable trap
X=X[:,1:]
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt=np.array(X[:,[0,1,2,3,4,5]], dtype=float)

#building the model using backward elimination
import statsmodels.api as sm
X_opt=np.array(X[:,[0,1,2,3,4,5]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() #x2 has the highest P value. Dropping [2].R2=0.951, adj R2=0.945


X_opt=np.array(X[:,[0,1,3,4,5]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() #x1 has higher P value. Dropping [1]. R2=0.951, AdjR2=0.946


X_opt=np.array(X[:,[0,3,4,5]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() #x2 has higher P value. Dropping [4]. R2=0.951, Adj R2=0.948


X_opt=np.array(X[:,[0,3,5]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() #X2 has higher P value. Dropping [5]. R2=0.950, Adj R2=0.948


X_opt=np.array(X[:,[0,3]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() # R&D is statistically significant to predict the results.R2=0.947, Adj R2=0.945


# Splitting the MLR into training and test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_opt_Train, X_opt_Test = train_test_split(X_opt,test_size = 0.2, random_state = 0)

#Fitting MLR into the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_opt_Train, y_Train)

# Predicting the Optimal Test set results
y_Optimal_Pred = regressor.predict(X_opt_Test)

#comparing y_optimal_pred with y_test
np.set_printoptions(precision=2)
print(np.concatenate((y_Optimal_Pred.reshape(len(y_Optimal_Pred),1),
                      y_Test.reshape(len(y_Test),1)),1))
#The predicted results are very much close to the test results. We can proceed with this model for predicting the profit. 
