import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from tensorflow.keras.models import Model, load_model

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("../../data/weather_prediction_dataset.csv")

# Select 8 Features

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

# ==========================
# Normalize
# ==========================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# ==========================
# Load Trained Autoencoder
# ==========================

autoencoder = load_model(
    "../../models/weather_autoencoder.keras"
)

# ==========================
# Create Encoder
# ==========================

encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.layers[2].output
)

# ==========================
# Extract Encoded Features
# ==========================

encoded_features = encoder.predict(
    X_scaled
)

print("\nEncoded Features Shape:")
print(encoded_features.shape)

# ==========================
# K-Means Clustering
# ==========================

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(
    encoded_features
)

# Add Cluster Column

df["Cluster"] = clusters

# ==========================
# Results
# ==========================

print("\nCluster Distribution:")

print(
    df["Cluster"].value_counts()
)

print("\nCluster Statistics:")

print(
    df.groupby("Cluster")[
        [
            "BASEL_cloud_cover",
            "BASEL_humidity",
            "BASEL_pressure",
            "BASEL_precipitation",
            "BASEL_temp_mean"
        ]
    ].mean()
)


# ==========================
# Visualization
# ==========================

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    encoded_features[:, 0],
    encoded_features[:, 1],
    c=clusters
)

plt.title(
    "Weather Pattern Clusters"
)

plt.xlabel(
    "Encoded Feature 1"
)

plt.ylabel(
    "Encoded Feature 2"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.savefig(
    "../../outputs/weather_clusters.png"
)

plt.show()