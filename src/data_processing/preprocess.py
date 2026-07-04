import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("../../data/weather_prediction_dataset.csv")

# Select 8 BASEL features
X = df[
    [
        "BASEL_cloud_cover",
        "BASEL_humidity",
        "BASEL_pressure",
        "BASEL_global_radiation",
        "BASEL_precipitation",
        "BASEL_sunshine",
        "BASEL_temp_mean",
        "BASEL_temp_max"
    ]
]

# Check for missing values
print("Missing Values:")
print(X.isnull().sum())

# Normalize
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print("\nDataset Shape:")
print(X.shape)

print("\nScaled Shape:")
print(X_scaled.shape)

print("\nFirst 5 Scaled Records:")
print(X_scaled[:5])