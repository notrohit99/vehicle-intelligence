import os
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.data_loader import (
    create_feature_bundle,
    create_sliding_windows,
)
from models.stgcn_model import SimpleSTGCN


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

H5_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "METR-LA.h5",
)

ADJ_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "adj_METR-LA.pkl",
)

WEATHER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "weather_CA_2019.csv",
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stgcn_weights.pt",
)


app = FastAPI(
    title="Vehicle Intelligence Traffic Service",
    description="ST-GCN traffic prediction service",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrafficModel:

    def __init__(self):
        self.bundle = None
        self.model = None
        self.edge_index = None

    def load(self):

        print("Loading METR-LA data...")

        self.bundle = create_feature_bundle(
            H5_PATH,
            ADJ_PATH,
            weather_path=WEATHER_PATH,
        )

        self.edge_index = torch.tensor(
            self.bundle.edge_index,
            dtype=torch.long,
        )

        in_features = self.bundle.features.shape[-1]

        print(f"Sensors: {len(self.bundle.sensor_ids)}")
        print(f"Features: {in_features}")

        self.model = SimpleSTGCN(
            in_features=in_features,
            hidden_size=32,
            horizon=3,
        )

        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Model weights not found: {WEIGHTS_PATH}"
            )

        self.model.load_state_dict(
            torch.load(
                WEIGHTS_PATH,
                map_location="cpu",
            )
        )

        self.model.eval()

        print("Traffic model loaded successfully.")


    def predict(self) -> Dict[str, float]:

        if self.model is None:
            raise RuntimeError(
                "Traffic model is not loaded."
            )

        features = self.bundle.features

        window = min(12, features.shape[0])

        x, _ = create_sliding_windows(
            features,
            self.bundle.target,
            window=window,
            horizon=1,
        )

        if len(x) == 0:
            raise ValueError(
                "Not enough data for prediction."
            )

        x_tensor = torch.tensor(
            x[-1:],
            dtype=torch.float32,
        )

        with torch.no_grad():

            prediction = self.model(
                x_tensor,
                self.edge_index,
            )

        prediction = (
            prediction[0, :, 0]
            .cpu()
            .numpy()
        )

        return {
            self.bundle.sensor_ids[i]: float(
                prediction[i]
            )
            for i in range(
                len(self.bundle.sensor_ids)
            )
        }


traffic_model = TrafficModel()


@app.on_event("startup")
def startup():
    traffic_model.load()


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "traffic",
        "model": "SimpleSTGCN",
        "sensors": (
            len(traffic_model.bundle.sensor_ids)
            if traffic_model.bundle
            else 0
        ),
    }


@app.post("/predict")
def predict():

    try:

        predictions = traffic_model.predict()

        return {
            "status": "success",
            "sensor_count": len(predictions),
            "predictions": predictions,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )