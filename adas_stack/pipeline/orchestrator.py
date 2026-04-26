from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

from ..common.metrics import MetricsRecorder
from ..common.vehicle_detection_dataset import VehicleDetectionDatasetAdapter, VehicleDetectionDatasetConfig
from ..common.schemas import EvaluationSnapshot, LatencySample
from ..common.simulation import SyntheticFrameConfig, SyntheticScene, SyntheticSceneGenerator
from .clients import HttpPipelineClient
from .streaming import AsyncStreamingPipeline


@dataclass
class FrameResult:
    frame_id: str
    risk_score: float
    severity: str


class PipelineOrchestrator:
    def __init__(self, frames: int, target_fps: int) -> None:
        self.frames = frames
        self.target_fps = target_fps
        self.generator = SyntheticSceneGenerator(SyntheticFrameConfig())
        self.metrics = MetricsRecorder()
        self.client = HttpPipelineClient()
        self.streaming_pipeline = AsyncStreamingPipeline()
        dataset_root = os.getenv("DATASET_ROOT")
        self.dataset_root = Path(dataset_root) if dataset_root else None

    async def run(self) -> list[FrameResult]:
        transport = os.getenv("PIPELINE_TRANSPORT", "http").lower()
        if transport == "queue":
            return await self._run_queue_pipeline()

        results: list[FrameResult] = []
        frame_interval = 1.0 / max(self.target_fps, 1)
        scenes = self._load_scenes()
        for scene in scenes:
            cycle_start = time.perf_counter()
            frame = scene.frame

            perception_start = time.perf_counter()
            observed_scene = await self.client.analyze(frame)
            self.metrics.record_latency(
                LatencySample(stage="perception", frame_id=frame.frame_id, latency_ms=(time.perf_counter() - perception_start) * 1000.0)
            )

            prediction_start = time.perf_counter()
            prediction_batch = await self.client.forecast(frame, observed_scene)
            self.metrics.record_latency(
                LatencySample(stage="prediction", frame_id=frame.frame_id, latency_ms=(time.perf_counter() - prediction_start) * 1000.0)
            )

            risk_start = time.perf_counter()
            assessment = await self.client.score(frame, observed_scene, prediction_batch)
            self.metrics.record_latency(
                LatencySample(stage="risk", frame_id=frame.frame_id, latency_ms=(time.perf_counter() - risk_start) * 1000.0)
            )

            detection_accuracy = self._estimate_detection_accuracy(scene.objects, observed_scene.detections)
            prediction_ade, prediction_brier = self._estimate_prediction_metrics(scene.objects, prediction_batch)
            self.metrics.record_detection_accuracy(detection_accuracy)
            self.metrics.record_prediction_error(prediction_ade, prediction_brier)
            self.metrics.mark_frame()

            results.append(FrameResult(frame_id=frame.frame_id, risk_score=assessment.risk_score, severity=assessment.severity))

            elapsed = time.perf_counter() - cycle_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
        await self.client.close()
        return results

    async def _run_queue_pipeline(self) -> list[FrameResult]:
        results: list[FrameResult] = []
        stream_results = await self.streaming_pipeline.run(self._load_scenes())
        for item in stream_results:
            detection_accuracy = self._estimate_detection_accuracy(item.ground_truth, item.observation.detections)
            prediction_ade, prediction_brier = self._estimate_prediction_metrics(
                item.ground_truth,
                SimpleNamespace(predictions=item.predictions),
            )
            self.metrics.record_detection_accuracy(detection_accuracy)
            self.metrics.record_prediction_error(prediction_ade, prediction_brier)
            self.metrics.mark_frame()
            self.metrics.record_latency(LatencySample(stage="perception", frame_id=item.frame.frame_id, latency_ms=item.perception_latency_ms))
            self.metrics.record_latency(LatencySample(stage="prediction", frame_id=item.frame.frame_id, latency_ms=item.prediction_latency_ms))
            self.metrics.record_latency(LatencySample(stage="risk", frame_id=item.frame.frame_id, latency_ms=item.risk_latency_ms))
            results.append(FrameResult(frame_id=item.frame.frame_id, risk_score=item.assessment.risk_score, severity=item.assessment.severity))
        return results

    def _estimate_detection_accuracy(self, ground_truth: list, detections: list) -> float:
        if not ground_truth:
            return 1.0
        matched = 0
        for truth in ground_truth:
            for detection in detections:
                if truth.object_class.value == detection.object_class.value:
                    truth_bbox = truth.bbox
                    det_bbox = detection.bbox
                    overlap = max(0.0, min(truth_bbox.x2, det_bbox.x2) - max(truth_bbox.x1, det_bbox.x1)) * max(
                        0.0, min(truth_bbox.y2, det_bbox.y2) - max(truth_bbox.y1, det_bbox.y1)
                    )
                    if overlap > 0:
                        matched += 1
                        break
        return matched / len(ground_truth)

    def _estimate_prediction_metrics(self, ground_truth: list, batch) -> tuple[float, float]:
        if not batch.predictions:
            return 0.0, 0.0
        errors: list[float] = []
        briers: list[float] = []
        for prediction in batch.predictions:
            candidates = [obj for obj in ground_truth if obj.object_class == prediction.object_class]
            if not candidates:
                continue
            predicted_center = prediction.trajectory[-1]
            truth = min(
                candidates,
                key=lambda obj: ((obj.bbox.center[0] - predicted_center.x) ** 2 + (obj.bbox.center[1] - predicted_center.y) ** 2),
            )
            if truth is None:
                continue
            truth_terminal_x = truth.bbox.center[0] + truth.velocity_x * (len(prediction.trajectory) * 0.2)
            truth_terminal_y = truth.bbox.center[1] + truth.velocity_y * (len(prediction.trajectory) * 0.2)
            errors.append(((predicted_center.x - truth_terminal_x) ** 2 + (predicted_center.y - truth_terminal_y) ** 2) ** 0.5)
            expected_collision = 1.0 if truth.velocity_y > 0 or abs(truth.velocity_x) > 6 else 0.0
            briers.append((prediction.collision_probability - expected_collision) ** 2)
        if not errors:
            return 0.0, 0.0
        return sum(errors) / len(errors), sum(briers) / len(briers)

    def _load_scenes(self) -> list[SyntheticScene]:
        if self.dataset_root is not None and self.dataset_root.exists():
            try:
                adapter = VehicleDetectionDatasetAdapter(VehicleDetectionDatasetConfig(self.dataset_root))
                scenes = list(adapter.iter_scenes())
                if scenes:
                    return scenes[: self.frames]
            except FileNotFoundError:
                pass
        return [self.generator.generate(sequence_id) for sequence_id in range(self.frames)]

    def snapshot(self) -> EvaluationSnapshot:
        return self.metrics.snapshot()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the ADAS streaming pipeline simulation")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=12)
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(frames=args.frames, target_fps=args.target_fps)
    results = await orchestrator.run()
    snapshot = orchestrator.snapshot().model_dump()
    print(json.dumps({"results": [item.__dict__ for item in results], "metrics": snapshot}, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
