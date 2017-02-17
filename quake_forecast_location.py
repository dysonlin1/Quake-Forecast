# coding: utf-8
# Machine Learning: Quake Forecast Location
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('quake_signals.csv')
print(train)

features = ['Duration', 'Height']
data = train[features]
print(data)

target = train['Location']
print(target)

model = RandomForestClassifier()
model.fit(data, target)
quake_location = model.predict([[2, 2.3]])
print(quake_location)