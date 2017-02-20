# coding: utf-8
# Machine Learning: Quake Forecast
import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv('quake_signals.csv')
print(train)

features = ['Duration', 'Height']
data = train[features]
print(data)

target = train['Time']
print(target)

model = LinearRegression()
model.fit(data, target)
quake_time = model.predict([[2, 2.3]])
print(quake_time)