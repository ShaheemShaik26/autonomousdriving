from __future__ import annotations

import os

from fastapi import FastAPI

from ...common.schemas import FrameEnvelope
from .core import PerceptionService


app = FastAPI(title="ADAS Perception Service", version="0.1.0")
service = PerceptionService(checkpoint_path=os.getenv("PERCEPTION_CHECKPOINT"))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/perception/analyze")
async def analyze(frame: FrameEnvelope) -> dict[str, object]:
    result = await service.analyze(frame)
    return {
        "scene": result.scene.model_dump(),
        "model_latency_ms": result.model_latency_ms,
    }
