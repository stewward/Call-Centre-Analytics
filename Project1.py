import pandas as pd
import numpy as np

##Step 2: Load/Gather Data
#Load CSV file into DataFrame
df = pd.read_csv('simulated_call_centre.csv')

#Check first rows
print(df.head())

##Step 3 Cleaning Data
#Check for missing values
print(df.isnull().sum())
#Check data types 
print(df.dtypes)
#Basic statistics
print(df.describe(include='all'))
#Convert 'Call Date' to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

##Explore Trends 
#Daily call volume
daily_calls = df.groupby('date').size()
print(daily_calls)
# #Plot daily call volume
daily_calls.plot(title='Daily Call Volume', ylabel='Number of Calls', xlabel='Date')        
import matplotlib.pyplot as plt
plt.show()  

#Average call duration by call type
avg_duration_per_day = df.groupby('date')['service_length'].mean()
print(avg_duration_per_day)
#Plot average call duration by call type
plt.figure(figsize=(10,5))
plt.plot(avg_duration_per_day.index, avg_duration_per_day.values)
plt.title("Average Service Length per Day")
plt.xlabel("Date")
plt.ylabel("Service Length (seconds/minutes)")
plt.show()  

##Forcasting and Predicted Modeling 
#Predicting daily call volume using simple model 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Fit model
model = ExponentialSmoothing(daily_calls, seasonal='add', seasonal_periods=7)
fit_model = model.fit()
#Forecast next 30 days
forecast = fit_model.forecast(30)

#Plot forecast
plt.figure(figsize=(10,5))
plt.plot(daily_calls.index, daily_calls.values, label='Historical')
plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
plt.title("Call Volume Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Calls")
plt.legend()
plt.show()

#Predict Service Length waiting time using Linear Regression
from sklearn.linear_model import LinearRegression
#Prepare data
X = df[['wait_length']]
y = df['service_length']
reg=LinearRegression().fit(X, y)
print("R^2:", reg.score(X, y))
