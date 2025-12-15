import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

DATA_PATH = "data/processed/dataset_v1_clean.csv"
OUTPUT_DIR = "data/processed"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["trip_duration"])
y = df["trip_duration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

mlflow.set_experiment("nyc_taxi_trip_duration")

with mlflow.start_run(run_name="train_test_split"):
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("train_rows", X_train.shape[0])
    mlflow.log_metric("test_rows", X_test.shape[0])

print("Train-test split complete and logged to MLflow")
