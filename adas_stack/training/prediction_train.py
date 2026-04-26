from __future__ import annotations

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.synthetic_tracking import SyntheticTrackingDataset, SyntheticTrackingDatasetConfig
from ..common.schemas import ObjectClass
from ..services.prediction_service.core import TrajectoryForecaster


@dataclass(frozen=True)
class PredictionTrainingConfig:
    data_root: Path | None
    output_path: Path
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    history_window: int = 4


class SyntheticTrackingForecastDataset(Dataset):
    def __init__(self, history_window: int = 4) -> None:
        self.adapter = SyntheticTrackingDataset(SyntheticTrackingDatasetConfig(history_window=history_window))
        self.history_window = history_window
        self.samples = self._build_samples()

    def _build_samples(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for history, target in self.adapter.iter_scene_windows(window_size=self.history_window):
            feature_sequence = []
            for record in history:
                feature_sequence.append([
                    record.bbox.center[0],
                    record.bbox.center[1],
                    record.bbox.width,
                    record.bbox.height,
                    float(record.track_id),
                    self._class_index(record.object_class),
                ])
            if len(feature_sequence) != self.history_window - 1:
                continue
            delta_x = target.bbox.center[0] - history[-1].bbox.center[0]
            delta_y = target.bbox.center[1] - history[-1].bbox.center[1]
            target_vector = torch.tensor(
                [delta_x, delta_y, target.bbox.width, target.bbox.height, 1.0 if target.occluded > 0 else 0.0],
                dtype=torch.float32,
            )
            samples.append((torch.tensor(np.asarray(feature_sequence), dtype=torch.float32), target_vector))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]

    def _class_index(self, object_class: ObjectClass) -> float:
        mapping = {ObjectClass.vehicle: 0.0, ObjectClass.pedestrian: 1.0, ObjectClass.obstacle: 2.0}
        return mapping[object_class]


def train(config: PredictionTrainingConfig) -> None:
    dataset = SyntheticTrackingForecastDataset(history_window=config.history_window)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    model = TrajectoryForecaster()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    displacement_loss_fn = nn.SmoothL1Loss()
    collision_loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(config.epochs):
        for feature_batch, target_batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(feature_batch)
            displacement_loss = displacement_loss_fn(outputs[:, :4], target_batch[:, :4])
            collision_loss = collision_loss_fn(outputs[:, 4], target_batch[:, 4])
            loss = displacement_loss + collision_loss
            loss.backward()
            optimizer.step()

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ADAS trajectory forecaster from synthetic motion samples")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--history-window", type=int, default=4)
    args = parser.parse_args()

    train(
        PredictionTrainingConfig(
            data_root=Path(args.data_root) if args.data_root else None,
            output_path=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            history_window=args.history_window,
        )
    )


if __name__ == "__main__":
    main()
