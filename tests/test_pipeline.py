from __future__ import annotations

import asyncio

from adas_stack.common.simulation import SyntheticSceneGenerator
from adas_stack.services.perception_service.core import PerceptionService
from adas_stack.services.prediction_service.core import PredictionService
from adas_stack.services.risk_service.core import RiskService


def test_end_to_end_in_memory_pipeline() -> None:
    scene = SyntheticSceneGenerator().generate(0)
    perception = PerceptionService()
    prediction = PredictionService()
    risk = RiskService()

    observed = asyncio.run(perception.analyze(scene.frame))
    forecast = asyncio.run(prediction.forecast(scene.frame, observed.scene))
    assessment = asyncio.run(risk.evaluate(scene.frame, observed.scene, forecast.batch.predictions))

    assert observed.scene.detections
    assert len(forecast.batch.predictions) == len(observed.scene.detections)
    assert 0.0 <= assessment.assessment.risk_score <= 1.0
