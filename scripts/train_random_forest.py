import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Categorical encoding
categorical_cols = ["store_and_fwd_flag"]

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 🔐 SAVE FEATURE ORDER (CRITICAL)
feature_columns = X_train.columns.tolist()
with open("feature_columns.txt", "w") as f:
    for col in feature_columns:
        f.write(col + "\n")

mlflow.set_experiment("nyc_taxi_trip_duration")

with mlflow.start_run(run_name="random_forest_baseline"):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 15)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Log model + feature schema
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("feature_columns.txt")

    print("Random Forest trained successfully")
    print("MAE:", mae)
    print("R2:", r2)
