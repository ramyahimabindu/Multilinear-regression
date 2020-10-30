import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data preprocessing
toyo=pd.read_csv('C:\\Users\\91950\\Downloads\\ToyotaCorolla.csv',encoding='latin1') #since there are some special characters, using latin1
toyo.columns=toyo.columns.str.replace ('_','') #replacing the special characters
toyo.drop (['Model','MfgMonth','MfgYear','FuelType','MetColor','Color','Automatic','Cylinders','MfrGuarantee', 'BOVAGGuarantee','GuaranteePeriod','ABS','Airbag1','Airbag2','Airco','Automaticairco','Boardcomputer','CDPlayer','CentralLock','PoweredWindows','PowerSteering','Radio','Mistlamps','SportModel','BackseatDivider','MetallicRim','Radiocassette','TowBar'],axis=1,inplace=True)
x=toyo.iloc[:,2:].values
y=toyo.iloc[:,1].values

#checking for missing values
toyo.isnull() 

#splitting the data into training and test set
from sklearn.model_selection import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size = 0.2, random_state = 0)

toyo.corr() #there are negative correlations

#fitting MLR into the training set and building a model directly
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit (x_Train,y_Train)

#predicting the test set results
y_pred=regression.predict(x_Test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),
                      y_Test.reshape(len(y_Test),1)),1)) #the results are pretty close. 

#building a model with backward elimination
import statsmodels.api as sm
x = np.append(arr = np. ones((1436, 1)).astype(int), values = x, axis = 1)
x_opt=np.array(x[:,[0,1,2,3,4,5,6,7,8]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary() #x5 has high p value.R2=0.864, Adj R2=0.863

#dropping x5 and building a model
x_opt=np.array(x[:,[0,1,2,3,4,6,7,8]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary() #R2=0.864, Adj R2=0.863. X4 Pvalue is higher than 0.05.

#dropping x4 and building a model
x_opt=np.array(x[:,[0,1,2,3,6,7,8]], dtype=float)
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary() #all values are statistically significant

#splitting the new model into training and test
X_opt_Train, X_opt_Test = train_test_split(x_opt,test_size = 0.2, random_state = 0)
regressor=LinearRegression()
regressor.fit(X_opt_Train, y_Train)

# Predicting the Optimal Test set results
y_Optimal_Pred = regressor.predict(X_opt_Test)

#predicting the test set results
np.set_printoptions(precision=2)
print(np.concatenate((y_Optimal_Pred.reshape(len(y_Optimal_Pred),1),
                      y_Test.reshape(len(y_Test),1)),1))
#the predicted values are very closer to the test set values. Since there is a negative correlation with the price factor, there is an indication that some of the other factors are inversely proportioanl w.r.t price. 

print(regressor.coef_) 
print (regressor.intercept_) #-6634.39393
y=-6634.3939-(1.22e+)
