from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from ..common.schemas import FrameEnvelope, GroundTruthObject, RiskAssessment, SceneObservation, SyntheticScene, TrajectoryPrediction
from ..services.perception_service.core import PerceptionService
from ..services.prediction_service.core import PredictionService
from ..services.risk_service.core import RiskService


@dataclass(frozen=True)
class StreamConfig:
    max_queue_depth: int = 8
    emit_interval_ms: int = 33


@dataclass(frozen=True)
class StreamResult:
    frame: FrameEnvelope
    ground_truth: list[GroundTruthObject]
    observation: SceneObservation
    predictions: list[TrajectoryPrediction]
    assessment: RiskAssessment
    perception_latency_ms: float
    prediction_latency_ms: float
    risk_latency_ms: float


class AsyncStreamingPipeline:
    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        self.perception = PerceptionService()
        self.prediction = PredictionService()
        self.risk = RiskService()

    async def run(self, scenes: list[SyntheticScene]) -> list[StreamResult]:
        source_queue: asyncio.Queue[tuple[FrameEnvelope, list[GroundTruthObject]] | None] = asyncio.Queue(maxsize=self.config.max_queue_depth)
        perception_queue: asyncio.Queue[tuple[FrameEnvelope, list[GroundTruthObject], SceneObservation, float] | None] = asyncio.Queue(maxsize=self.config.max_queue_depth)
        prediction_queue: asyncio.Queue[tuple[FrameEnvelope, list[GroundTruthObject], SceneObservation, list, float, float] | None] = asyncio.Queue(maxsize=self.config.max_queue_depth)
        output_queue: asyncio.Queue[StreamResult | None] = asyncio.Queue(maxsize=self.config.max_queue_depth)

        async def producer() -> None:
            for scene in scenes:
                await source_queue.put((scene.frame, scene.objects))
                await asyncio.sleep(self.config.emit_interval_ms / 1000.0)
            await source_queue.put(None)

        async def perception_worker() -> None:
            while True:
                item = await source_queue.get()
                if item is None:
                    await perception_queue.put(None)
                    return
                frame, ground_truth = item
                start = time.perf_counter()
                result = await self.perception.analyze(frame)
                await perception_queue.put((frame, ground_truth, result.scene, (time.perf_counter() - start) * 1000.0))

        async def prediction_worker() -> None:
            while True:
                item = await perception_queue.get()
                if item is None:
                    await prediction_queue.put(None)
                    return
                frame, ground_truth, scene, perception_latency_ms = item
                start = time.perf_counter()
                result = await self.prediction.forecast(frame, scene)
                prediction_latency_ms = (time.perf_counter() - start) * 1000.0
                await prediction_queue.put((frame, ground_truth, scene, result.batch.predictions, perception_latency_ms, prediction_latency_ms))

        async def risk_worker() -> None:
            while True:
                item = await prediction_queue.get()
                if item is None:
                    await output_queue.put(None)
                    return
                frame, ground_truth, scene, predictions, perception_latency_ms, prediction_latency_ms = item
                start = time.perf_counter()
                result = await self.risk.evaluate(frame, scene, predictions)
                risk_latency_ms = (time.perf_counter() - start) * 1000.0
                await output_queue.put(
                    StreamResult(
                        frame=frame,
                        ground_truth=ground_truth,
                        observation=scene,
                        predictions=predictions,
                        assessment=result.assessment,
                        perception_latency_ms=perception_latency_ms,
                        prediction_latency_ms=prediction_latency_ms,
                        risk_latency_ms=risk_latency_ms,
                    )
                )

        results: list[StreamResult] = []

        async def collector() -> None:
            while True:
                item = await output_queue.get()
                if item is None:
                    return
                results.append(item)

        await asyncio.gather(
            producer(),
            perception_worker(),
            prediction_worker(),
            risk_worker(),
            collector(),
        )
        return results
