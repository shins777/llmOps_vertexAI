from fastapi import FastAPI, Request

import joblib
import json
import numpy as np
import pickle
import os

from google.cloud import storage
from sklearn.datasets import load_iris

app = FastAPI()
gcs_client = storage.Client()

print(f"---------[ Environment ] --------------")
print(f"PORT:{os.environ['AIP_HTTP_PORT']}")
print(f"AIP_STORAGE_URI:{os.environ['AIP_STORAGE_URI']}")
print(f"AIP_HEALTH_ROUTE:{os.environ['AIP_HEALTH_ROUTE']}")
print(f"AIP_PREDICT_ROUTE:{os.environ['AIP_PREDICT_ROUTE']}")

# Download model file from GCS
with open("model.pkl", 'wb') as model_f:
    gcs_client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/model.pkl", model_f
    )

# Load model file stored in local was downloaded from GCS
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

_class_names = load_iris().target_names
_model = model

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    print(f"Health Check : OK !!")

    return {"status":"OK"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    print(f"Prediction request body : {body}")

    instances = body["instances"]
    inputs = np.asarray(instances)
    outputs = _model.predict(inputs)

    return {"predictions": [_class_names[class_num] for class_num in outputs]}
