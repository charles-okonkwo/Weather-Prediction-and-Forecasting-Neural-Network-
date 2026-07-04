import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load dataset
df = pd.read_csv("../../data/weather_prediction_dataset.csv")

# Select features
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

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(
    scaler,
    "../../models/scaler.pkl"
)

# Split data
X_train, X_test = train_test_split(
    X_scaled,
    test_size=0.3,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Build Autoencoder
input_layer = Input(shape=(8,))

hidden1 = Dense(6, activation='relu')(input_layer)

bottleneck = Dense(3, activation='linear')(hidden1)

hidden2 = Dense(6, activation='relu')(bottleneck)

output_layer = Dense(8, activation='sigmoid')(hidden2)

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train
history = autoencoder.fit(
    X_train,
    X_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, X_test)
)

# Save history
import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "../../outputs/training_history.csv",
    index=False
)
# Save Model
autoencoder.save(
    "../../models/weather_autoencoder.keras"
)

print("\nModel saved successfully!")