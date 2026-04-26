from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ObjectClass(str, Enum):
    vehicle = "vehicle"
    pedestrian = "pedestrian"
    obstacle = "obstacle"


class FrameEnvelope(BaseModel):
    frame_id: str
    timestamp_ms: int
    sequence_id: int
    width: int
    height: int
    encoded_image: str
    source: Literal["synthetic", "kitti", "bdd100k", "vehicle-detection", "simulation"] = "synthetic"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class ObjectState(BaseModel):
    object_id: str
    object_class: ObjectClass
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    velocity_x: float
    velocity_y: float
    track_age: int = 0


class SceneObservation(BaseModel):
    frame_id: str
    timestamp_ms: int
    detections: list[ObjectState]
    object_density: float
    camera_health: Literal["ok", "degraded", "failed"] = "ok"


class TrajectoryPoint(BaseModel):
    timestamp_ms: int
    x: float
    y: float


class TrajectoryPrediction(BaseModel):
    object_id: str
    object_class: ObjectClass
    trajectory: list[TrajectoryPoint]
    collision_probability: float = Field(ge=0.0, le=1.0)
    risk_contribution: float = Field(ge=0.0, le=1.0)


class PredictionBatch(BaseModel):
    frame_id: str
    timestamp_ms: int
    predictions: list[TrajectoryPrediction]
    model_latency_ms: float


class RiskAssessment(BaseModel):
    frame_id: str
    timestamp_ms: int
    risk_score: float = Field(ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high", "critical"]
    contributing_objects: list[str]
    explanation: str


class LatencySample(BaseModel):
    stage: str
    frame_id: str
    latency_ms: float


class EvaluationSnapshot(BaseModel):
    frames_processed: int
    detection_accuracy: float
    prediction_ade: float
    prediction_collision_brier: float
    avg_latency_ms: dict[str, float]


@dataclass(frozen=True)
class GroundTruthObject:
    object_id: str
    object_class: ObjectClass
    bbox: BoundingBox
    velocity_x: float
    velocity_y: float


@dataclass(frozen=True)
class SyntheticScene:
    frame: FrameEnvelope
    objects: list[GroundTruthObject]


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))
