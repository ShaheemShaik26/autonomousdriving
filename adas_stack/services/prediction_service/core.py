from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ...common.schemas import FrameEnvelope, ObjectState, PredictionBatch, SceneObservation, TrajectoryPoint, TrajectoryPrediction, clamp_probability


class TrajectoryForecaster(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1, :])


@dataclass
class PredictionResult:
    batch: PredictionBatch


class PredictionService:
    def __init__(self, checkpoint_path: str | None = None, horizon_steps: int = 8, step_ms: int = 200) -> None:
        self.model = TrajectoryForecaster()
        self.model.eval()
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state)
        self.horizon_steps = horizon_steps
        self.step_ms = step_ms

    async def forecast(self, frame: FrameEnvelope, scene: SceneObservation) -> PredictionResult:
        start = asyncio.get_running_loop().time()
        predictions = [self._predict_object(frame, detection, scene) for detection in scene.detections]
        latency_ms = (asyncio.get_running_loop().time() - start) * 1000.0
        batch = PredictionBatch(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            predictions=predictions,
            model_latency_ms=latency_ms,
        )
        return PredictionResult(batch=batch)

    def _predict_object(self, frame: FrameEnvelope, detection: ObjectState, scene: SceneObservation) -> TrajectoryPrediction:
        sequence = self._build_sequence(detection)
        with torch.no_grad():
            output = self.model(sequence)
        delta_x, delta_y, _, _, collision_logit = output.squeeze(0).tolist()
        center_x, center_y = detection.bbox.center
        velocity_x = detection.velocity_x + delta_x
        velocity_y = detection.velocity_y + delta_y
        trajectory: list[TrajectoryPoint] = []
        future_probability = 0.0
        for step in range(1, self.horizon_steps + 1):
            x = center_x + velocity_x * (step * self.step_ms / 1000.0)
            y = center_y + velocity_y * (step * self.step_ms / 1000.0)
            trajectory.append(TrajectoryPoint(timestamp_ms=frame.timestamp_ms + step * self.step_ms, x=x, y=y))
            if y > frame.height * 0.72 or abs(x - frame.width / 2.0) < frame.width * 0.1:
                future_probability = max(future_probability, 0.45 + scene.object_density * 0.4)
        collision_probability = clamp_probability(torch.sigmoid(torch.tensor(collision_logit)).item() * 0.55 + future_probability)
        risk_contribution = clamp_probability(0.35 * scene.object_density + 0.65 * collision_probability)
        return TrajectoryPrediction(
            object_id=detection.object_id,
            object_class=detection.object_class,
            trajectory=trajectory,
            collision_probability=collision_probability,
            risk_contribution=risk_contribution,
        )

    def _build_sequence(self, detection: ObjectState) -> torch.Tensor:
        center_x, center_y = detection.bbox.center
        features = np.array(
            [[
                center_x,
                center_y,
                detection.bbox.width,
                detection.bbox.height,
                detection.velocity_x,
                detection.velocity_y,
            ]],
            dtype=np.float32,
        )
        repeated = np.repeat(features[None, :, :], repeats=4, axis=1)
        return torch.from_numpy(repeated)
