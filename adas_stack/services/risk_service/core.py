from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...common.schemas import FrameEnvelope, RiskAssessment, SceneObservation, TrajectoryPrediction, clamp_probability


@dataclass
class RiskResult:
    assessment: RiskAssessment


class RiskEngine:
    def score(self, frame: FrameEnvelope, scene: SceneObservation, predictions: list[TrajectoryPrediction]) -> RiskAssessment:
        risk_components: list[float] = []
        contributors: list[str] = []
        ego_x = frame.width / 2.0
        ego_y = frame.height * 0.7

        for prediction in predictions:
            terminal_point = prediction.trajectory[-1]
            distance = ((terminal_point.x - ego_x) ** 2 + (terminal_point.y - ego_y) ** 2) ** 0.5
            distance_component = clamp_probability(1.0 - min(distance / max(frame.width, frame.height), 1.0))
            if scene.detections:
                motion_magnitude = sum(abs(obj.velocity_x) + abs(obj.velocity_y) for obj in scene.detections) / len(scene.detections)
            else:
                motion_magnitude = 0.0
            velocity_component = clamp_probability(motion_magnitude / 30.0)
            density_component = scene.object_density
            overlap_component = prediction.collision_probability
            risk = clamp_probability(0.35 * distance_component + 0.2 * velocity_component + 0.2 * density_component + 0.25 * overlap_component)
            risk_components.append(risk * max(0.2, prediction.risk_contribution))
            if risk > 0.35:
                contributors.append(prediction.object_id)

        if risk_components:
            raw_score = sum(risk_components) / len(risk_components)
        else:
            raw_score = scene.object_density * 0.25

        if raw_score >= 0.8:
            severity = "critical"
        elif raw_score >= 0.6:
            severity = "high"
        elif raw_score >= 0.35:
            severity = "medium"
        else:
            severity = "low"

        explanation = (
            f"Risk computed from {len(predictions)} forecasts with density {scene.object_density:.2f}, "
            f"trajectory overlap and terminal distance to ego corridor."
        )
        return RiskAssessment(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            risk_score=clamp_probability(raw_score),
            severity=severity,
            contributing_objects=contributors,
            explanation=explanation,
        )


class RiskService:
    def __init__(self) -> None:
        self.engine = RiskEngine()

    async def evaluate(self, frame: FrameEnvelope, scene: SceneObservation, predictions: list[TrajectoryPrediction]) -> RiskResult:
        await asyncio.sleep(0)
        assessment = self.engine.score(frame, scene, predictions)
        return RiskResult(assessment=assessment)
