from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..common.vehicle_detection_dataset import VehicleDetectionDatasetAdapter, VehicleDetectionDatasetConfig
from ..common.schemas import ObjectClass
from ..services.perception_service.core import TinyObjectClassifier


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output_path: Path
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3


class VehicleDetectionCropDataset(Dataset):
    def __init__(self, config: VehicleDetectionDatasetConfig) -> None:
        self.adapter = VehicleDetectionDatasetAdapter(config)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[tuple[torch.Tensor, int]]:
        samples: list[tuple[torch.Tensor, int]] = []
        for sample in self.adapter.iter_samples():
            with Image.open(sample.image_path) as opened_image:
                image = opened_image.convert("RGB")
                for annotation in sample.annotations:
                    crop = image.crop((annotation.bbox.x1, annotation.bbox.y1, annotation.bbox.x2, annotation.bbox.y2))
                    tensor = self._preprocess(crop)
                    samples.append((tensor, self._class_index(annotation.object_class)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.samples[index]

    def _preprocess(self, crop: Image.Image) -> torch.Tensor:
        resized = crop.resize((32, 32))
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return tensor

    def _class_index(self, object_class: ObjectClass) -> int:
        mapping = {ObjectClass.vehicle: 0, ObjectClass.pedestrian: 1, ObjectClass.obstacle: 2}
        return mapping[object_class]


def train(config: TrainingConfig) -> None:
    dataset = VehicleDetectionCropDataset(VehicleDetectionDatasetConfig(config.data_root))
    if not dataset.samples:
        raise RuntimeError("No vehicle-detection crops were found for training.")

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    model = TinyObjectClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.epochs):
        for batch_images, batch_labels in dataloader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ADAS perception classifier from Kaggle vehicle-detection crops")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        TrainingConfig(
            data_root=Path(args.data_root),
            output_path=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    )


if __name__ == "__main__":
    main()
