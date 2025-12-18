from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import mlflow
import mlflow.pyfunc
import pandas as pd
import os

# -----------------------------
# App configuration
# -----------------------------
app = FastAPI(
    title="ML Inference Service",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# -----------------------------
# Security (API Key Auth)
# -----------------------------
security = HTTPBearer()
API_KEY = os.getenv("API_KEY")

def authenticate(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured"
        )

    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

# -----------------------------
# MLflow configuration
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "nyc_taxi_rf")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# Globals
# -----------------------------
model = None
feature_columns = None

# -----------------------------
# Load model + schema on startup
# -----------------------------
@app.on_event("startup")
def startup_load():
    global model, feature_columns

    try:
        client = mlflow.tracking.MlflowClient()

        # Resolve model version via alias
        mv = client.get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS
        )

        # Download model artifacts locally
        local_model_path = client.download_artifacts(
            mv.run_id, "model"
        )

        model = mlflow.pyfunc.load_model(local_model_path)

        # Download feature schema
        feature_path = client.download_artifacts(
            mv.run_id, "feature_columns.txt"
        )

        with open(feature_path) as f:
            feature_columns = [line.strip() for line in f]

        print("✅ Model & feature schema loaded successfully")

    except Exception as e:
        print("❌ Startup load failed:", e)

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ml-inference",
        "model_loaded": model is not None
    }

# -----------------------------
# Prediction endpoint (PROTECTED)
# -----------------------------
@app.post("/predict")
def predict(
    payload: dict,
    _: None = Depends(authenticate)
):
    if model is None or feature_columns is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    df = pd.DataFrame([payload])

    if "store_and_fwd_flag" in df.columns:
        df = pd.get_dummies(
            df,
            columns=["store_and_fwd_flag"],
            drop_first=True
        )

    # Align schema
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    prediction = model.predict(df)[0]
    return {"trip_duration_prediction": float(prediction)}
