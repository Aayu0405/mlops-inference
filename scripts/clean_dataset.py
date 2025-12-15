import pandas as pd
import numpy as np

RAW_PATH = "data/raw/dataset_v1.csv"
PROCESSED_PATH = "data/processed/dataset_v1_clean.csv"

df = pd.read_csv(RAW_PATH)

# Convert datetime columns
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["trip_distance_km"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove invalid records
df = df[df["trip_duration"] > 0]
df = df[df["passenger_count"] > 0]

# Feature engineering
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_day"] = df["pickup_datetime"].dt.dayofweek

# Drop unused columns
df = df.drop(columns=["id", "pickup_datetime", "dropoff_datetime"])

df.to_csv(PROCESSED_PATH, index=False)

print("Clean dataset saved to:", PROCESSED_PATH)
print("Final shape:", df.shape)
