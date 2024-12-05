import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error,accuracy_score
from sklearn.svm import SVR



df = pd.read_csv("stock.csv")
df.drop(columns=['Unnamed: 0'],inplace=True)
# print(df.head())
# print(df.shape)     # size of the data....
# print(df.info())    # datatypes of data...
# print(df.describe())   # Stats od data...
# print(df.isnull().sum())  # find null valuessss...
# df.set_index('Date',inplace=True)

# df.Close.plot(figsize=(10,7), color='r')
# plt.ylabel("Prices")
# plt.xlabel("Date")
# plt.title("Price Series")
# plt.show()
df.drop(columns=['Date'], inplace=True)

# sns.distplot(df["Close"])
# sns.distplot(df["Open"])
# sns.distplot(df["High"])
# plt.show()

x =df.drop(columns=['Close'], axis=1)
y =df['Close']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=20,random_state=42)

# lr = LinearRegression().fit(x_train,y_train)
# la = Lasso().fit(x_train,y_train)
rd =Ridge().fit(x_train,y_train)
pred = rd.predict(x_test)

def calculate_metrics(y_test,y_pred):
    mse = mean_squared_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print("MSE:",mse)
    print("RMSE:",rmse)
    print("R2_Score:",r2)

# calculate_metrics(y_test,pred)

svm = SVR(C=10,gamma=0.01,kernel='rbf')
svm.fit(x_train,y_train)
svm_pred = svm.predict(x_test)

calculate_metrics(y_test,svm_pred)