# coding: utf-8
# Machine Learning: Quake Forecast Magnitude
import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv('quake_signals.csv')
print(train)

features = ['Duration', 'Height']
data = train[features]
print(data)

target = train['Magnitude']
print(target)

model = LinearRegression()
model.fit(data, target)
quake_magnitude = model.predict([[2, 2.3]])
print(quake_magnitude)