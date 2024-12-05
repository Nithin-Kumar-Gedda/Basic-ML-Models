import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from prophet import Prophet
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

df = pd.read_csv('Traffic data.csv')
df.drop(columns=['ID'], inplace=True)
# print(df.head())
# print(df.isnull().sum())

# convert object int0 datetime attribute
df['Datetime']=pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')

# plt.plot(df['Datetime'], df['Count'])
# plt.show()
df.index = df['Datetime']
df['y'] = df['Count']
df.drop(columns=['Datetime','Count'], axis=1, inplace=True)
df= df.resample('D').sum()
df['ds'] = df.index

size = 60

train , test = train_test_split(df, test_size=size/len(df), shuffle=False)

model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(train)

future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)
# print(forcast.tail())
# print(test.tail())
# model.plot_components(forcast)
# plt.show()

pred = forecast.iloc[-60:, :]
# plt.plot(test['ds'], test['y'])
# plt.plot(pred['ds'], pred['yhat'], color= 'red')
# plt.show()

# print("Train head:")
print(train.head())
print("Test tail:")
print(test.tail())
# print("Future DataFrame head:")
# print(future.head())
# print("Future DataFrame tail:")
# print(future.tail())
print("Forecast tail:")
print(forecast.tail(60))
# print("Pred tail:")
# print(pred.tail())