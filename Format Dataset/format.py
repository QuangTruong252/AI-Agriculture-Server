from datetime import datetime
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('D:\Học tập\Edge-Container-AI\dos\city_temperature.csv')
print(df)

newData = df[["Year","Day", "Month", "AvgTemperature"]]
print(newData)

imputer = SimpleImputer()

imputer.fit(newData[["AvgTemperature"]])

newData[["AvgTemperature"]] = imputer.transform(newData[["AvgTemperature"]])

print(newData)

newData[["AvgTemperature"]].info()

newData["Datetime"] = pd.to_datetime(newData[["Year","Day","Month"]],format='%d-%m-%Y', errors='coerce')
newData = newData.drop_duplicates(subset='Datetime',keep='last')
print('******************newData had drop_duplicates******************')
print(newData)

del newData["Year"]
del newData["Day"]
del newData["Month"]

print(newData)

newData[["Datetime"]].info()
newData[["AvgTemperature"]].info()

newData.to_csv("Output.csv")

