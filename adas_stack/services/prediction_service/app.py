from __future__ import annotations

import os

from fastapi import FastAPI

from ...common.schemas import FrameEnvelope, SceneObservation
from .core import PredictionService


app = FastAPI(title="ADAS Prediction Service", version="0.1.0")
service = PredictionService(checkpoint_path=os.getenv("PREDICTION_CHECKPOINT"))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/prediction/forecast")
async def forecast(payload: dict[str, object]) -> dict[str, object]:
    frame = FrameEnvelope.model_validate(payload["frame"])
    scene = SceneObservation.model_validate(payload["scene"])
    result = await service.forecast(frame, scene)
    return {"batch": result.batch.model_dump()}
