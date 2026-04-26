from __future__ import annotations

from dataclasses import dataclass
import base64
import csv
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schemas import BoundingBox, FrameEnvelope, GroundTruthObject, ObjectClass, SyntheticScene


@dataclass(frozen=True)
class KittiLikeDatasetConfig:
    root: Path


@dataclass(frozen=True)
class KittiObjectAnnotation:
    object_id: str
    object_class: ObjectClass
    bbox: BoundingBox
    truncated: float
    occluded: int
    alpha: float
    confidence: float = 1.0


@dataclass(frozen=True)
class KittiFrameSample:
    frame: FrameEnvelope
    annotations: list[KittiObjectAnnotation]


_CLASS_ALIASES: dict[str, ObjectClass] = {
    "car": ObjectClass.vehicle,
    "van": ObjectClass.vehicle,
    "truck": ObjectClass.vehicle,
    "tram": ObjectClass.vehicle,
    "pedestrian": ObjectClass.pedestrian,
    "person_sitting": ObjectClass.pedestrian,
    "cyclist": ObjectClass.pedestrian,
    "misc": ObjectClass.obstacle,
    "dontcare": ObjectClass.obstacle,
}


class KittiLikeDatasetAdapter:
    """Minimal adapter for KITTI-style layouts.

    Expected layout:
    - image_2/*.png
    - label_2/*.txt
    - velodyne/ optionally ignored for this reference implementation
    """

    def __init__(self, config: KittiLikeDatasetConfig) -> None:
        self.config = config

    def iter_frames(self) -> list[FrameEnvelope]:
        return [sample.frame for sample in self.iter_samples()]

    def iter_samples(self) -> list[KittiFrameSample]:
        image_dir = self.config.root / "image_2"
        label_dir = self.config.root / "label_2"
        samples: list[KittiFrameSample] = []

        for index, image_path in enumerate(sorted(image_dir.glob("*"))):
            width, height = self._image_size(image_path)
            frame = FrameEnvelope(
                frame_id=image_path.stem,
                timestamp_ms=index * 33,
                sequence_id=index,
                width=width,
                height=height,
                encoded_image=base64.b64encode(image_path.read_bytes()).decode("ascii"),
                source="kitti",
            )
            annotations = self._load_annotations(label_dir / f"{image_path.stem}.txt")
            samples.append(KittiFrameSample(frame=frame, annotations=annotations))

        return samples

    def iter_scenes(self) -> Iterator[SyntheticScene]:
        for sample in self.iter_samples():
            yield SyntheticScene(frame=sample.frame, objects=self._annotations_to_truth(sample.annotations))

    def _image_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size

    def _load_annotations(self, label_path: Path) -> list[KittiObjectAnnotation]:
        if not label_path.exists():
            return []

        annotations: list[KittiObjectAnnotation] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for index, raw_line in enumerate(handle):
                parts = raw_line.strip().split()
                if len(parts) < 15:
                    continue
                class_name = parts[0].lower()
                object_class = _CLASS_ALIASES.get(class_name, ObjectClass.obstacle)
                truncated = float(parts[1])
                occluded = int(float(parts[2]))
                alpha = float(parts[3])
                x1, y1, x2, y2 = map(float, parts[4:8])
                annotations.append(
                    KittiObjectAnnotation(
                        object_id=f"{label_path.stem}_{index}",
                        object_class=object_class,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        truncated=truncated,
                        occluded=occluded,
                        alpha=alpha,
                    )
                )
        return annotations

    def _annotations_to_truth(self, annotations: list[KittiObjectAnnotation]) -> list[GroundTruthObject]:
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
