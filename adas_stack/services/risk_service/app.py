from __future__ import annotations

from fastapi import FastAPI

from ...common.schemas import FrameEnvelope, PredictionBatch, SceneObservation
from .core import RiskService


app = FastAPI(title="ADAS Risk Service", version="0.1.0")
service = RiskService()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/risk/score")
async def score(payload: dict[str, object]) -> dict[str, object]:
    frame = FrameEnvelope.model_validate(payload["frame"])
    scene = SceneObservation.model_validate(payload["scene"])
    batch = PredictionBatch.model_validate(payload["batch"])
    result = await service.evaluate(frame, scene, batch.predictions)
    return {"assessment": result.assessment.model_dump()}
