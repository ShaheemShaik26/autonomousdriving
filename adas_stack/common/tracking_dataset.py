from __future__ import annotations

from dataclasses import dataclass
import base64
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schemas import BoundingBox, FrameEnvelope, GroundTruthObject, ObjectClass


@dataclass(frozen=True)
class TrackingObjectRecord:
    frame_index: int
    track_id: int
    object_class: ObjectClass
    bbox: BoundingBox
    truncated: float
    occluded: int
    alpha: float


@dataclass(frozen=True)
class TrackingSequenceSample:
    sequence_id: str
    frame: FrameEnvelope
    objects: list[TrackingObjectRecord]


@dataclass(frozen=True)
class KittiTrackingDatasetConfig:
    root: Path


_TRACKING_CLASS_ALIASES: dict[str, ObjectClass] = {
    "car": ObjectClass.vehicle,
    "van": ObjectClass.vehicle,
    "truck": ObjectClass.vehicle,
    "tram": ObjectClass.vehicle,
    "pedestrian": ObjectClass.pedestrian,
    "person": ObjectClass.pedestrian,
    "person_sitting": ObjectClass.pedestrian,
    "cyclist": ObjectClass.pedestrian,
    "misc": ObjectClass.obstacle,
    "dontcare": ObjectClass.obstacle,
}


class KittiTrackingDatasetAdapter:
    """Read KITTI tracking sequences from the public benchmark layout."""

    def __init__(self, config: KittiTrackingDatasetConfig) -> None:
        self.config = config

    def iter_sequences(self) -> Iterator[list[TrackingSequenceSample]]:
        image_root = self.config.root / "image_02"
        label_root = self.config.root / "label_02"
        if not image_root.exists():
            return

        for sequence_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            sequence_id = sequence_dir.name
            label_path = label_root / f"{sequence_id}.txt"
            records = self._load_records(label_path)
            samples: list[TrackingSequenceSample] = []
            for frame_path in sorted(sequence_dir.glob("*.png")):
                frame_index = int(frame_path.stem)
                width, height = self._image_size(frame_path)
                frame = FrameEnvelope(
                    frame_id=f"{sequence_id}_{frame_index:06d}",
                    timestamp_ms=frame_index * 100,
                    sequence_id=frame_index,
                    width=width,
                    height=height,
                    encoded_image=base64.b64encode(frame_path.read_bytes()).decode("ascii"),
                    source="kitti",
                )
                frame_objects = [record for record in records if record.frame_index == frame_index]
                samples.append(TrackingSequenceSample(sequence_id=sequence_id, frame=frame, objects=frame_objects))
            yield samples

    def iter_scene_windows(self, window_size: int = 5) -> Iterator[tuple[list[TrackingObjectRecord], TrackingObjectRecord]]:
        for sequence_samples in self.iter_sequences() or []:
            tracks: dict[int, list[TrackingObjectRecord]] = {}
            for sample in sequence_samples:
                for record in sample.objects:
                    tracks.setdefault(record.track_id, []).append(record)
            for track_records in tracks.values():
                ordered = sorted(track_records, key=lambda item: item.frame_index)
                if len(ordered) < window_size:
                    continue
                for index in range(window_size - 1, len(ordered)):
                    history = ordered[index - window_size + 1 : index]
                    target = ordered[index]
                    if len(history) == window_size - 1:
                        yield history, target

    def to_truth(self, record: TrackingObjectRecord) -> GroundTruthObject:
        return GroundTruthObject(
            object_id=f"track_{record.track_id}",
            object_class=record.object_class,
            bbox=record.bbox,
            velocity_x=0.0,
            velocity_y=0.0,
        )

    def _load_records(self, label_path: Path) -> list[TrackingObjectRecord]:
        if not label_path.exists():
            return []

        records: list[TrackingObjectRecord] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split()
                if len(parts) < 17:
                    continue
                frame_index = int(parts[0])
                track_id = int(parts[1])
                object_class = _TRACKING_CLASS_ALIASES.get(parts[2].lower(), ObjectClass.obstacle)
                truncated = float(parts[3])
                occluded = int(float(parts[4]))
                alpha = float(parts[5])
                x1, y1, x2, y2 = map(float, parts[6:10])
                records.append(
                    TrackingObjectRecord(
                        frame_index=frame_index,
                        track_id=track_id,
                        object_class=object_class,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        truncated=truncated,
                        occluded=occluded,
                        alpha=alpha,
                    )
                )
        return records

    def _image_size(self, frame_path: Path) -> tuple[int, int]:
        with Image.open(frame_path) as image:
            return image.size
