import joblib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# ===========================================
# Load Dataset
# ===========================================

df = pd.read_csv("../../data/weather_prediction_dataset.csv")

# ===========================================
# Select Features
# ===========================================

features = [
    "BASEL_cloud_cover",
    "BASEL_humidity",
    "BASEL_pressure",
    "BASEL_global_radiation",
    "BASEL_precipitation",
    "BASEL_sunshine",
    "BASEL_temp_mean",
    "BASEL_temp_max"
]

X = df[features]

# ===========================================
# Load Scaler
# ===========================================

scaler = joblib.load("../../models/scaler.pkl")

X_scaled = scaler.transform(X)

# ===========================================
# Create Forecast Dataset
# ===========================================

X_input = X_scaled[:-1]
y_output = X_scaled[1:]

# ===========================================
# Train/Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_input,
    y_output,
    test_size=0.2,
    shuffle=False
)

# ===========================================
# Build ANN
# ===========================================

model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(8,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(8))

# ===========================================
# Compile
# ===========================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ===========================================
# Train
# ===========================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ===========================================
# Evaluate
# ===========================================

loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nForecast Model Results")
print("----------------------")
print("Loss :", loss)
print("MAE  :", mae)

# ===========================================
# Save Model
# ===========================================

model.save("../../models/weather_forecast.keras")

print("\nForecast model saved successfully!")


# ===========================================
# Forecast Learning Curve
# ===========================================

plt.figure(figsize=(10,6))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Forecast ANN Learning Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("../outputs/forecast_learning_curve.png")

plt.show()