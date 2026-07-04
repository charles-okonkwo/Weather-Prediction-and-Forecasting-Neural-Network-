import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# ===========================================
# Load Forecast Model
# ===========================================

model = load_model("../../models/weather_forecast.keras")

# ===========================================
# Load Scaler
# ===========================================

scaler = joblib.load("../../models/scaler.pkl")

# ===========================================
# Load Dataset
# ===========================================

df = pd.read_csv("../../data/weather_prediction_dataset.csv")

# ===========================================
# Weather Features
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

# ===========================================
# Learn Weather Statistics from Dataset
# ===========================================

avg_temp = df["BASEL_temp_mean"].mean()
avg_humidity = df["BASEL_humidity"].mean()
avg_cloud = df["BASEL_cloud_cover"].mean()
avg_sunshine = df["BASEL_sunshine"].mean()

light_rain = df["BASEL_precipitation"].quantile(0.25)
heavy_rain = df["BASEL_precipitation"].quantile(0.75)

# ===========================================
# Get Last Day in Dataset
# ===========================================

last_day = df[features].iloc[-1:]

current = scaler.transform(last_day)

predictions = []

# ===========================================
# Predict Next 3 Days
# ===========================================

for day in range(3):

    next_day = model.predict(current, verbose=0)

    prediction = scaler.inverse_transform(next_day)

    predictions.append(prediction[0])

    current = next_day

# ===========================================
# Convert to DataFrame
# ===========================================

forecast = pd.DataFrame(
    predictions,
    columns=features,
    index=["Day 1", "Day 2", "Day 3"]
)

# ===========================================
# Save Forecast
# ===========================================

forecast.to_csv("../../outputs/three_day_forecast.csv", index=True)

# ===========================================
# Display Forecast
# ===========================================

print("\n")
print("=" * 65)
print("             WEATHER FORECAST REPORT")
print("=" * 65)

for i in range(len(forecast)):

    row = forecast.iloc[i]

    cloud = row["BASEL_cloud_cover"]
    humidity = row["BASEL_humidity"]
    pressure = row["BASEL_pressure"]
    radiation = row["BASEL_global_radiation"]
    rainfall = row["BASEL_precipitation"]
    sunshine = row["BASEL_sunshine"]
    mean_temp = row["BASEL_temp_mean"]
    max_temp = row["BASEL_temp_max"]

    print(f"\nForecast for Day {i+1}")
    print("-" * 45)

    print(f"Mean Temperature : {mean_temp:.2f} °C")
    print(f"Maximum Temp     : {max_temp:.2f} °C")
    print(f"Humidity         : {humidity:.2f} %")
    print(f"Pressure         : {pressure:.2f} hPa")
    print(f"Cloud Cover      : {cloud:.2f}")
    print(f"Sunshine         : {sunshine:.2f} hrs")
    print(f"Rainfall         : {rainfall:.2f} mm")
    print(f"Solar Radiation  : {radiation:.2f}")

    print("\nWeather Analysis")
    print("-" * 25)

    # Heavy Rain
    if rainfall >= heavy_rain:

        print("Heavy rainfall is expected.")
        print("The sky will remain cloudy.")
        print("Carry an umbrella if going outside.")
        print("Outdoor activities are not recommended.")

    # Light Rain
    elif rainfall >= light_rain:

        print("Light rainfall is expected.")
        print("Cloudy conditions may persist.")
        print("A raincoat or umbrella is advisable.")

    # Sunny
    elif sunshine > avg_sunshine and cloud < avg_cloud:

        if mean_temp > avg_temp:

            print("Warm and sunny weather is expected.")
            print("Clear skies are likely throughout the day.")
            print("Excellent weather for outdoor activities.")

        else:

            print("Sunny weather is expected.")
            print("Pleasant weather conditions are expected.")

    # Cloudy
    elif cloud > avg_cloud:

        if humidity > avg_humidity:

            print("Cloudy and humid weather is expected.")
            print("The weather may feel warmer than usual.")

        else:

            print("Mostly cloudy weather is expected.")
            print("Little sunshine is expected.")

    # Humid
    elif humidity > avg_humidity:

        print("Humid weather is expected.")
        print("You may experience warm conditions.")

    # Warm
    elif mean_temp > avg_temp:

        print("A warm day is expected.")
        print("Weather conditions should remain stable.")

    # Cool
    else:

        print("Cool and stable weather is expected.")
        print("No significant weather changes are anticipated.")

print("\n" + "=" * 65)
print("Forecast completed successfully.")
print("Results saved to:")
print("../../outputs/three_day_forecast.csv")
print("=" * 65)