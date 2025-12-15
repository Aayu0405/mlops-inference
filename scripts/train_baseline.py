import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Explicitly define low-cardinality categorical columns
categorical_cols = ["store_and_fwd_flag"]

# One-hot encode ONLY categorical columns
X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    drop_first=True
)

# Align columns between train and test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# MLflow experiment
mlflow.set_experiment("nyc_taxi_trip_duration")

with mlflow.start_run(run_name="linear_regression_baseline"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(model, "model")

    print("Baseline model trained successfully")
    print("MAE:", mae)
    print("R2:", r2)
