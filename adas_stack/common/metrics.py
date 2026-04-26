from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .schemas import EvaluationSnapshot, LatencySample


@dataclass
class RunningMetric:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


class MetricsRecorder:
    def __init__(self, history_size: int = 512) -> None:
        self.history_size = history_size
        self.latency_by_stage: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history_size))
        self.detection_accuracy = RunningMetric()
        self.prediction_ade = RunningMetric()
        self.prediction_collision_brier = RunningMetric()
        self.frames_processed = 0

    def record_latency(self, sample: LatencySample) -> None:
        self.latency_by_stage[sample.stage].append(sample.latency_ms)

    def record_detection_accuracy(self, value: float) -> None:
        self.detection_accuracy.update(value)

    def record_prediction_error(self, ade: float, brier: float) -> None:
        self.prediction_ade.update(ade)
        self.prediction_collision_brier.update(brier)

    def mark_frame(self) -> None:
        self.frames_processed += 1

    def snapshot(self) -> EvaluationSnapshot:
        average_latency = {
            stage: (sum(values) / len(values) if values else 0.0)
            for stage, values in self.latency_by_stage.items()
        }
        return EvaluationSnapshot(
            frames_processed=self.frames_processed,
            detection_accuracy=self.detection_accuracy.average,
            prediction_ade=self.prediction_ade.average,
            prediction_collision_brier=self.prediction_collision_brier.average,
            avg_latency_ms=average_latency,
        )
