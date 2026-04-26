from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ..common.schemas import BoundingBox, FrameEnvelope, ObjectClass, ObjectState, SceneObservation, GroundTruthObject
from ..common.synthetic_tracking import SyntheticTrackingDataset, SyntheticTrackingDatasetConfig
from ..services.prediction_service.core import PredictionService


@dataclass(frozen=True)
class PredictionEvaluationResult:
    frames_evaluated: int
    mean_ade: float
    mean_fde: float
    mean_collision_brier: float
    classwise_ade: dict[str, float]


def evaluate_tracking_dataset(data_root: Path | None, checkpoint_path: Path | None = None) -> PredictionEvaluationResult:
    adapter = SyntheticTrackingDataset(SyntheticTrackingDatasetConfig(history_window=4))
    service = PredictionService(checkpoint_path=str(checkpoint_path) if checkpoint_path else None)

    ade_values: list[float] = []
    fde_values: list[float] = []
    collision_brier_values: list[float] = []
    per_class_errors: dict[str, list[float]] = {cls.value: [] for cls in ObjectClass}

    frames_evaluated = 0
    for history, target in adapter.iter_scene_windows(window_size=4):
        frames_evaluated += 1
        last_history = history[-1]
        prev_history = history[-2]
        frame_width = max(int(target.bbox.x2 + 20), 640)
        frame_height = max(int(target.bbox.y2 + 20), 360)
        delta_seconds = max((target.frame_index - last_history.frame_index) * 0.1, 0.1)
        velocity_x = (last_history.bbox.center[0] - prev_history.bbox.center[0]) / max(delta_seconds, 1e-6)
        velocity_y = (last_history.bbox.center[1] - prev_history.bbox.center[1]) / max(delta_seconds, 1e-6)

        pseudo_frame = FrameEnvelope(
            frame_id=f"track_{target.track_id}_{target.frame_index}",
            timestamp_ms=target.frame_index * 100,
            sequence_id=target.frame_index,
            width=frame_width,
            height=frame_height,
            encoded_image="",
            source="simulation",
        )
        scene = SceneObservation(
            frame_id=pseudo_frame.frame_id,
            timestamp_ms=pseudo_frame.timestamp_ms,
            detections=[
                ObjectState(
                    object_id=f"track_{target.track_id}",
                    object_class=target.object_class,
                    confidence=1.0,
                    bbox=BoundingBox(
                        x1=last_history.bbox.x1,
                        y1=last_history.bbox.y1,
                        x2=last_history.bbox.x2,
                        y2=last_history.bbox.y2,
                    ),
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    track_age=1,
                )
            ],
            object_density=0.0,
            camera_health="ok",
        )

        prediction = service._predict_object(pseudo_frame, scene.detections[0], scene)
        terminal = prediction.trajectory[-1]
        ground_truth_x = target.bbox.center[0]
        ground_truth_y = target.bbox.center[1]
        ade = sum(
            ((point.x - ground_truth_x) ** 2 + (point.y - ground_truth_y) ** 2) ** 0.5
            for point in prediction.trajectory
        ) / len(prediction.trajectory)
        fde = ((terminal.x - ground_truth_x) ** 2 + (terminal.y - ground_truth_y) ** 2) ** 0.5
        expected_collision = 1.0 if target.occluded > 0 else 0.0

        ade_values.append(ade)
        fde_values.append(fde)
        collision_brier_values.append((prediction.collision_probability - expected_collision) ** 2)
        per_class_errors[target.object_class.value].append(ade)

    classwise = {
        class_name: (sum(values) / len(values) if values else 0.0)
        for class_name, values in per_class_errors.items()
    }
    return PredictionEvaluationResult(
        frames_evaluated=frames_evaluated,
        mean_ade=(sum(ade_values) / len(ade_values)) if ade_values else 0.0,
        mean_fde=(sum(fde_values) / len(fde_values)) if fde_values else 0.0,
        mean_collision_brier=(sum(collision_brier_values) / len(collision_brier_values)) if collision_brier_values else 0.0,
        classwise_ade=classwise,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the synthetic trajectory forecaster")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    result = evaluate_tracking_dataset(Path(args.data_root) if args.data_root else None, Path(args.checkpoint) if args.checkpoint else None)
    print(json.dumps({
        "frames_evaluated": result.frames_evaluated,
        "mean_ade": result.mean_ade,
        "mean_fde": result.mean_fde,
        "mean_collision_brier": result.mean_collision_brier,
        "classwise_ade": result.classwise_ade,
    }, indent=2))


if __name__ == "__main__":
    main()
