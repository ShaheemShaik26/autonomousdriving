from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..evaluation.perception_eval import evaluate_dataset as evaluate_perception
from ..evaluation.prediction_eval import evaluate_tracking_dataset as evaluate_prediction
from ..training.perception_train import TrainingConfig as PerceptionTrainingConfig, train as train_perception
from ..training.prediction_train import PredictionTrainingConfig, train as train_prediction


@dataclass(frozen=True)
class BootstrapConfig:
    data_root: Path
    tracking_root: Path | None
    output_dir: Path
    frames: int
    target_fps: int


def bootstrap_stack(config: BootstrapConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = config.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    perception_checkpoint = checkpoints_dir / "perception.pt"
    prediction_checkpoint = checkpoints_dir / "prediction.pt"

    train_perception(
        PerceptionTrainingConfig(
            data_root=config.data_root,
            output_path=perception_checkpoint,
        )
    )
    train_prediction(PredictionTrainingConfig(data_root=None, output_path=prediction_checkpoint))

    perception_metrics = evaluate_perception(config.data_root, perception_checkpoint)
    prediction_metrics = evaluate_prediction(None, prediction_checkpoint)

    env = os.environ.copy()
    env["PIPELINE_TRANSPORT"] = "queue"
    env["DATASET_ROOT"] = str(config.data_root)
    env["PERCEPTION_CHECKPOINT"] = str(perception_checkpoint)
    env["PREDICTION_CHECKPOINT"] = str(prediction_checkpoint)

    project_root = Path(__file__).resolve().parents[2]
    override_path = config.output_dir / "docker-compose.override.generated.yml"
    override_path.write_text(_compose_override_yaml(config), encoding="utf-8")

    compose_command = _compose_base_command() + [
        "-f",
        str(project_root / "docker-compose.yml"),
        "-f",
        str(override_path),
        "up",
        "--build",
    ]
    subprocess.run(compose_command, cwd=project_root, env=env, check=True)

    print(
        json.dumps(
            {
                "perception": perception_metrics.__dict__,
                "prediction": prediction_metrics.__dict__,
                "checkpoints": {
                    "perception": str(perception_checkpoint),
                    "prediction": str(prediction_checkpoint),
                },
                "compose_override": str(override_path),
            },
            indent=2,
        )
    )


def _compose_base_command() -> list[str]:
    return ["docker", "compose"] if sys.platform != "win32" else ["docker", "compose"]


def _compose_override_yaml(config: BootstrapConfig) -> str:
    checkpoints_dir = config.output_dir / "checkpoints"
    checkpoints_mount = _to_compose_path(checkpoints_dir)
    dataset_mount = _to_compose_path(config.data_root)
    tracking_mount = _to_compose_path(config.tracking_root or config.data_root)
    return f"""services:\n  perception-service:\n    volumes:\n      - {checkpoints_mount}:/app/checkpoints:ro\n    environment:\n      - PERCEPTION_CHECKPOINT=/app/checkpoints/perception.pt\n\n  prediction-service:\n    volumes:\n      - {checkpoints_mount}:/app/checkpoints:ro\n    environment:\n      - PREDICTION_CHECKPOINT=/app/checkpoints/prediction.pt\n\n  risk-service:\n    environment:\n      - PYTHONUNBUFFERED=1\n\n  orchestrator:\n    volumes:\n      - {checkpoints_mount}:/app/checkpoints:ro\n      - {dataset_mount}:/app/data:ro\n      - {tracking_mount}:/app/tracking:ro\n    environment:\n      - PERCEPTION_URL=http://perception-service:8001\n      - PREDICTION_URL=http://prediction-service:8002\n      - RISK_URL=http://risk-service:8003\n      - PIPELINE_TRANSPORT=queue\n      - DATASET_ROOT=/app/data\n"""


def _to_compose_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate, and launch the ADAS stack")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tracking-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--target-fps", type=int, default=15)
    args = parser.parse_args()

    bootstrap_stack(
        BootstrapConfig(
            data_root=Path(args.data_root),
            tracking_root=Path(args.tracking_root) if args.tracking_root else None,
            output_dir=Path(args.output_dir),
            frames=args.frames,
            target_fps=args.target_fps,
        )
    )


if __name__ == "__main__":
    main()
