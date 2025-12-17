from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import mlflow
import mlflow.pyfunc
import pandas as pd
import os
import time

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
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_PATH = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# -----------------------------
# Lazy model loading (PROD SAFE)
# -----------------------------
model = None

def load_model():
    global model
    if model is not None:
        return model

    try:
        model = mlflow.pyfunc.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Model not ready yet: {e}")
        return None

# -----------------------------
# Feature columns loading
# -----------------------------
FEATURE_COLUMNS_PATH = os.getenv(
    "FEATURE_COLUMNS_PATH",
    "/app/feature_columns.txt"
)

if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise RuntimeError(
        f"Feature columns file not found at {FEATURE_COLUMNS_PATH}"
    )

with open(FEATURE_COLUMNS_PATH) as f:
    FEATURE_COLUMNS = [line.strip() for line in f.readlines()]

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
    m = load_model()
    if m is None:
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

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLUMNS]

    prediction = m.predict(df)
    return {"trip_duration_prediction": float(prediction[0])}
