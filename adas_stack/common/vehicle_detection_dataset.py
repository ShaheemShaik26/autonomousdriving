from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schemas import BoundingBox, FrameEnvelope, GroundTruthObject, ObjectClass, SyntheticScene

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional dependency
    kagglehub = None


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class VehicleDetectionDatasetConfig:
    root: Path
    split: str = "train"


@dataclass(frozen=True)
class VehicleDetectionObjectAnnotation:
    object_id: str
    object_class: ObjectClass
    bbox: BoundingBox
    label: str
    confidence: float = 1.0


@dataclass(frozen=True)
class VehicleDetectionFrameSample:
    frame: FrameEnvelope
    image_path: Path
    annotations: list[VehicleDetectionObjectAnnotation]


def download_kaggle_vehicle_detection() -> Path:
    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed. Install kagglehub to download the Kaggle vehicle-detection dataset.")
    return Path(kagglehub.dataset_download("killa92/vehicle-detection-dataset"))


def resolve_vehicle_detection_root(root: Path | None = None) -> Path:
    if root is not None and root.exists():
        return root
    return download_kaggle_vehicle_detection()


class VehicleDetectionDatasetAdapter:
    def __init__(self, config: VehicleDetectionDatasetConfig) -> None:
        self.config = config

    def iter_frames(self) -> list[FrameEnvelope]:
        return [sample.frame for sample in self.iter_samples()]

    def iter_samples(self) -> list[VehicleDetectionFrameSample]:
        image_dir = self._find_image_dir()
        label_dir = self._find_label_dir()
        class_names = self._load_class_names()

        samples: list[VehicleDetectionFrameSample] = []
        image_paths = [path for path in sorted(image_dir.rglob("*")) if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES]
        for index, image_path in enumerate(image_paths):
            width, height = self._image_size(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            annotations = self._parse_label_file(label_path, image_path.stem, width, height, class_names)
            frame = FrameEnvelope(
                frame_id=image_path.stem,
                timestamp_ms=index * 100,
                sequence_id=index,
                width=width,
                height=height,
                encoded_image=base64.b64encode(image_path.read_bytes()).decode("ascii"),
                source="vehicle-detection",
            )
            samples.append(VehicleDetectionFrameSample(frame=frame, image_path=image_path, annotations=annotations))
        return samples

    def iter_scenes(self) -> Iterator[SyntheticScene]:
        for sample in self.iter_samples():
            yield SyntheticScene(frame=sample.frame, objects=self._annotations_to_truth(sample.annotations))

    def _find_image_dir(self) -> Path:
        candidates = [
            self.config.root / "images" / self.config.split,
            self.config.root / "images",
            self.config.root / self.config.split / "images",
            self.config.root / self.config.split,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find a vehicle image directory under {self.config.root}")

    def _find_label_dir(self) -> Path:
        candidates = [
            self.config.root / "labels" / self.config.split,
            self.config.root / "labels",
            self.config.root / self.config.split / "labels",
            self.config.root / self.config.split,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find a label directory under {self.config.root}")

    def _load_class_names(self) -> list[str]:
        class_file = self.config.root / "classes.txt"
        if not class_file.exists():
            return ["vehicle"]
        return [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _parse_label_file(
        self,
        label_path: Path,
        fallback_stem: str,
        image_width: int,
        image_height: int,
        class_names: list[str],
    ) -> list[VehicleDetectionObjectAnnotation]:
        if not label_path.exists():
            return []

        annotations: list[VehicleDetectionObjectAnnotation] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for index, raw_line in enumerate(handle):
                parts = raw_line.strip().replace(",", " ").split()
                if len(parts) < 4:
                    continue
                tokens = parts
                label_token = tokens[0] if len(tokens) > 4 else "vehicle"
                coordinate_tokens = tokens[-4:]
                bbox = self._decode_bbox(coordinate_tokens, image_width, image_height)
                if bbox is None:
                    continue
                object_class = ObjectClass.vehicle
                label = self._normalize_label(label_token, class_names)
                annotations.append(
                    VehicleDetectionObjectAnnotation(
                        object_id=f"{fallback_stem}_{index}",
                        object_class=object_class,
                        bbox=bbox,
                        label=label,
                    )
                )
        return annotations

    def _decode_bbox(self, coordinate_tokens: list[str], image_width: int, image_height: int) -> BoundingBox | None:
        try:
            values = [float(value) for value in coordinate_tokens]
        except ValueError:
            return None

        if max(values) <= 1.5:
            x_center, y_center, box_width, box_height = values
            x_center *= image_width
            y_center *= image_height
            box_width *= image_width
            box_height *= image_height
            x1 = max(0.0, x_center - box_width / 2.0)
            y1 = max(0.0, y_center - box_height / 2.0)
            x2 = min(float(image_width), x_center + box_width / 2.0)
            y2 = min(float(image_height), y_center + box_height / 2.0)
            return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + abs(x2)
            y2 = y1 + abs(y2)
        return BoundingBox(x1=max(0.0, x1), y1=max(0.0, y1), x2=min(float(image_width), x2), y2=min(float(image_height), y2))

    def _normalize_label(self, token: str, class_names: list[str]) -> str:
        if token.isdigit():
            index = int(token)
            if 0 <= index < len(class_names):
                return class_names[index]
        return token or "vehicle"

    def _annotations_to_truth(self, annotations: list[VehicleDetectionObjectAnnotation]) -> list[GroundTruthObject]:
        return [
            GroundTruthObject(
                object_id=item.object_id,
                object_class=item.object_class,
                bbox=item.bbox,
                velocity_x=0.0,
                velocity_y=0.0,
            )
            for item in annotations
        ]

    def _image_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size