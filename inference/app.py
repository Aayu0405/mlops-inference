from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import mlflow.pyfunc
import pandas as pd
import os

# -----------------------------
# App configuration (PROD SAFE)
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
# Model loading (LOCAL ARTIFACT)
# -----------------------------
MODEL_PATH = "mlruns/1/models/m-a222fd5ccf4945da8e51175009becfff/artifacts"

model = mlflow.pyfunc.load_model(MODEL_PATH)

# -----------------------------
# Feature order loading
# -----------------------------
with open("feature_columns.txt") as f:
    FEATURE_COLUMNS = [line.strip() for line in f.readlines()]


# -----------------------------
# Health check (K8s + LB)
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ml-inference",
        "model_loaded": True
    }


# -----------------------------
# Prediction endpoint (PROTECTED)
# -----------------------------
@app.post("/predict")
def predict(
    payload: dict,
    _: None = Depends(authenticate)
):
    df = pd.DataFrame([payload])

    # Same encoding as training
    if "store_and_fwd_flag" in df.columns:
        df = pd.get_dummies(df, columns=["store_and_fwd_flag"], drop_first=True)

    # Add missing columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Enforce correct order
    df = df[FEATURE_COLUMNS]

    prediction = model.predict(df)
    return {"trip_duration_prediction": float(prediction[0])}
