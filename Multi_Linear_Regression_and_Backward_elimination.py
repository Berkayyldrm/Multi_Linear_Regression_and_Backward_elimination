import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("play_Tennis.csv") 
print(data)

# Data Preprocessing

from sklearn import preprocessing

data2 = data.apply(preprocessing.LabelEncoder().fit_transform) # We converted all datas with LabelEncoder 
# But we must convert some data with OHE or don't convert. 

outlook = data.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray() # OHE 
print(outlook)

##############################################

outlookdf = pd.DataFrame(data=outlook, index= range(14), columns =["overcast","rainy","sunny"]) #dataframe 
sonveriler = pd.concat([outlookdf,data.iloc[:,1:3]], axis = 1)
sonveriler = pd.concat([sonveriler,data2.iloc[:,3:5]],axis = 1)


from sklearn.model_selection import train_test_split 
# play variable is depented variable, others are indepented variables.
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)#1.indepented variable 2.depented variable


# Multi Linear Regression Model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test) # Prediction

##################################################

# Backward Elimination 
   # Multi Linear Regressin Formula => Y = B0 + B1*x1 + B2*x2 + e

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis = 1) # We added a column matrix of 1's to the data for the B's in the formula.

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values 
X_l = np.array(X_l, dtype = float) 

model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit() # We reach statics of data with OLS # 1. variable depented variable, 2. indepented variables.
print(model.summary()) #  Backward elimination method is used here. The value with the highest P value is eliminated. (First, the SI value and the P value are compared.)

# The p-value of the temperature variable was eliminated because it was high.

X_l = sonveriler.iloc[:,[0,1,2,4,5]].values 
X_l = np.array(X_l, dtype = float) 
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit() 
print(model.summary())

# New Prediction

x_train.drop('temperature',axis=1, inplace=True)
x_test.drop('temperature',axis=1, inplace=True)

regressor.fit(x_train,y_train)

y_pred2 = regressor.predict(x_test)