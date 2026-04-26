from __future__ import annotations

import os

import httpx

from ..common.schemas import FrameEnvelope, PredictionBatch, SceneObservation, RiskAssessment


class ServiceEndpoints:
    def __init__(self) -> None:
        self.perception_url = os.getenv("PERCEPTION_URL", "http://127.0.0.1:8001")
        self.prediction_url = os.getenv("PREDICTION_URL", "http://127.0.0.1:8002")
        self.risk_url = os.getenv("RISK_URL", "http://127.0.0.1:8003")


class HttpPipelineClient:
    def __init__(self, endpoints: ServiceEndpoints | None = None) -> None:
        self.endpoints = endpoints or ServiceEndpoints()
        self.client = httpx.AsyncClient(timeout=10.0)

    async def analyze(self, frame: FrameEnvelope) -> SceneObservation:
        response = await self.client.post(f"{self.endpoints.perception_url}/v1/perception/analyze", json=frame.model_dump())
        response.raise_for_status()
        return SceneObservation.model_validate(response.json()["scene"])

    async def forecast(self, frame: FrameEnvelope, scene: SceneObservation) -> PredictionBatch:
        response = await self.client.post(
            f"{self.endpoints.prediction_url}/v1/prediction/forecast",
            json={"frame": frame.model_dump(), "scene": scene.model_dump()},
        )
        response.raise_for_status()
        return PredictionBatch.model_validate(response.json()["batch"])

    async def score(self, frame: FrameEnvelope, scene: SceneObservation, batch: PredictionBatch) -> RiskAssessment:
        response = await self.client.post(
            f"{self.endpoints.risk_url}/v1/risk/score",
            json={"frame": frame.model_dump(), "scene": scene.model_dump(), "batch": batch.model_dump()},
        )
        response.raise_for_status()
        return RiskAssessment.model_validate(response.json()["assessment"])

    async def close(self) -> None:
        await self.client.aclose()
