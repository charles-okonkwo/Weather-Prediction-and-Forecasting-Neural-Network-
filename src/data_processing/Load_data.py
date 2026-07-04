import pandas as pd

df = pd.read_csv("../../data/weather_prediction_dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 30 Columns:")
print(df.columns[:30])