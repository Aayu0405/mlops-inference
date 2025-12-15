import mlflow

RUN_ID = "8f422ce69fe9489ba77efe2f91b6a577"
MODEL_NAME = "nyc_taxi_rf"

model_uri = f"runs:/{RUN_ID}/model"

mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Model registered:", MODEL_NAME)
