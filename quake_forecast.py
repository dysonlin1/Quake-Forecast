
# coding: utf-8
# Quake Forecast: Predict Time, Location and Magnitude of a Quake
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

left_half_duration = 3
right_half_duration = 1
right_half_estimate = 2
predict_duration = left_half_duration + right_half_duration + right_half_estimate
predict_height = 8

train = pd.read_csv('quake_signals.csv')
print(train)
features = ['Duration', 'Height']
data = train[features]
print(data)

# Predict Time
target = train['Time']
print(target)
model = LinearRegression()
model.fit(data, target)

quake_time = model.predict([[predict_duration, predict_height]])
print(quake_time)
print('Time: ' , round(right_half_estimate + quake_time[0]), ' days')

# Predict Location
target = train['Location']
print(target)
model = RandomForestClassifier()
model.fit(data, target)

quake_location = model.predict([[predict_duration, predict_height]])
print(quake_location)

# Predict Magnitude
target = train['Magnitude']
print(target)
model = LinearRegression()
model.fit(data, target)

quake_magnitude = model.predict([[predict_duration, predict_height]])
print(quake_magnitude)



