from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schemas import BoundingBox, FrameEnvelope, GroundTruthObject, ObjectClass, SyntheticScene, bbox_iou

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional dependency
    kagglehub = None


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_CLASS_ALIASES: dict[str, ObjectClass] = {
    "car": ObjectClass.vehicle,
    "bus": ObjectClass.vehicle,
    "truck": ObjectClass.vehicle,
    "trailer": ObjectClass.vehicle,
    "train": ObjectClass.vehicle,
    "bike": ObjectClass.vehicle,
    "motor": ObjectClass.vehicle,
    "other vehicle": ObjectClass.vehicle,
    "pedestrian": ObjectClass.pedestrian,
    "person": ObjectClass.pedestrian,
    "rider": ObjectClass.pedestrian,
    "traffic light": ObjectClass.obstacle,
    "traffic sign": ObjectClass.obstacle,
}


@dataclass(frozen=True)
class Bdd100kDatasetConfig:
    root: Path
    split: str = "train"


@dataclass(frozen=True)
class Bdd100kObjectAnnotation:
    object_id: str
    object_class: ObjectClass
    bbox: BoundingBox
    category: str
    confidence: float = 1.0
    occluded: int = 0


@dataclass(frozen=True)
class Bdd100kFrameSample:
    frame: FrameEnvelope
    image_path: Path
    video_name: str
    frame_index: int
    annotations: list[Bdd100kObjectAnnotation]


@dataclass(frozen=True)
class Bdd100kTrackingRecord:
    track_id: int
    video_name: str
    frame_index: int
    object_class: ObjectClass
    bbox: BoundingBox
    occluded: int = 0
    confidence: float = 1.0


@dataclass(frozen=True)
class Bdd100kTrackingDatasetConfig:
    root: Path
    split: str = "train"
    iou_threshold: float = 0.3


def download_kaggle_bdd100k() -> Path:
    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed. Install kagglehub to download the public BDD100K mirror from Kaggle.")
    return Path(kagglehub.dataset_download("solesensei/solesensei_bdd100k"))


def resolve_bdd100k_root(root: Path | None = None) -> Path:
    if root is not None and root.exists():
        return root
    return download_kaggle_bdd100k()


class Bdd100kDatasetAdapter:
    def __init__(self, config: Bdd100kDatasetConfig) -> None:
        self.config = config

    def iter_frames(self) -> list[FrameEnvelope]:
        return [sample.frame for sample in self.iter_samples()]

    def iter_samples(self) -> list[Bdd100kFrameSample]:
        image_dir = self._find_image_dir()
        label_entries = self._load_label_entries()
        image_lookup = self._build_image_lookup(image_dir)

        samples: list[Bdd100kFrameSample] = []
        for index, image_name in enumerate(sorted(image_lookup)):
            image_path = image_lookup[image_name]
            label_entry = label_entries.get(image_name) or label_entries.get(image_path.stem)
            annotations = self._parse_annotations(label_entry, image_path.stem)
            width, height = self._image_size(image_path)
            frame_index = self._frame_index(label_entry, index)
            video_name = self._video_name(label_entry, image_path.stem)
            frame = FrameEnvelope(
                frame_id=f"{video_name}_{frame_index}",
                timestamp_ms=frame_index * 100,
                sequence_id=index,
                width=width,
                height=height,
                encoded_image=base64.b64encode(image_path.read_bytes()).decode("ascii"),
                source="bdd100k",
            )
            samples.append(
                Bdd100kFrameSample(
                    frame=frame,
                    image_path=image_path,
                    video_name=video_name,
                    frame_index=frame_index,
                    annotations=annotations,
                )
            )

        return samples

    def iter_scenes(self) -> Iterator[SyntheticScene]:
        for sample in self.iter_samples():
            yield SyntheticScene(frame=sample.frame, objects=self._annotations_to_truth(sample.annotations))

    def _find_image_dir(self) -> Path:
        candidates = [
            self.config.root / "bdd100k" / "images" / "100k" / self.config.split,
            self.config.root / "images" / "100k" / self.config.split,
            self.config.root / "bdd100k" / "images" / self.config.split,
            self.config.root / "images" / self.config.split,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        for candidate in self.config.root.rglob(self.config.split):
            if candidate.is_dir() and any(path.suffix.lower() in _IMAGE_SUFFIXES for path in candidate.iterdir() if path.is_file()):
                return candidate

        raise FileNotFoundError(f"Could not find a BDD100K image directory under {self.config.root}")

    def _build_image_lookup(self, image_dir: Path) -> dict[str, Path]:
        lookup: dict[str, Path] = {}
        for image_path in image_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in _IMAGE_SUFFIXES:
                lookup[image_path.name] = image_path
                lookup.setdefault(image_path.stem, image_path)
        return lookup

    def _load_label_entries(self) -> dict[str, dict]:
        entries: dict[str, dict] = {}
        for label_path in self._candidate_label_paths():
            try:
                payload = json.loads(label_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for record in self._extract_records(payload):
                name = str(record.get("name") or "")
                if not name:
                    continue
                entries[name] = record
                entries.setdefault(Path(name).stem, record)
        return entries

    def _candidate_label_paths(self) -> list[Path]:
        preferred_roots = [
            self.config.root / "bdd100k_labels_release",
            self.config.root / "bdd100k" / "labels",
            self.config.root / "labels",
            self.config.root,
        ]
        paths: list[Path] = []
        seen: set[Path] = set()
        for root in preferred_roots:
            if not root.exists():
                continue
            for label_path in root.rglob("*.json"):
                name = label_path.name.lower()
                if any(token in name for token in ("label", "bdd100k", "det", "annotations")) and label_path not in seen:
                    seen.add(label_path)
                    paths.append(label_path)
        return paths

    def _extract_records(self, payload: object) -> list[dict]:
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if isinstance(payload, dict):
            for key in ("frames", "images", "labels", "data", "annotations"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [record for record in value if isinstance(record, dict)]
            if "name" in payload:
                return [payload]
        return []

    def _parse_annotations(self, label_entry: dict | None, fallback_stem: str) -> list[Bdd100kObjectAnnotation]:
        if not label_entry:
            return []

        annotations: list[Bdd100kObjectAnnotation] = []
        for index, item in enumerate(label_entry.get("labels", [])):
            if not isinstance(item, dict):
                continue
            box2d = item.get("box2d")
            if not isinstance(box2d, dict):
                continue
            category = str(item.get("category", "")).strip().lower()
            object_class = _CLASS_ALIASES.get(category, ObjectClass.obstacle)
            attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
            annotations.append(
                Bdd100kObjectAnnotation(
                    object_id=f"{fallback_stem}_{index}",
                    object_class=object_class,
                    bbox=BoundingBox(
                        x1=float(box2d.get("x1", 0.0)),
                        y1=float(box2d.get("y1", 0.0)),
                        x2=float(box2d.get("x2", 0.0)),
                        y2=float(box2d.get("y2", 0.0)),
                    ),
                    category=category,
                    confidence=float(item.get("score", 1.0)),
                    occluded=int(float(attributes.get("occluded", 0))) if attributes else 0,
                )
            )
        return annotations

    def _annotations_to_truth(self, annotations: list[Bdd100kObjectAnnotation]) -> list[GroundTruthObject]:
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

    def _frame_index(self, label_entry: dict | None, fallback_index: int) -> int:
        if not label_entry:
            return fallback_index
        value = label_entry.get("frameIndex")
        return int(value) if value is not None else fallback_index

    def _video_name(self, label_entry: dict | None, fallback_stem: str) -> str:
        if not label_entry:
            return fallback_stem
        value = label_entry.get("videoName") or label_entry.get("video_name")
        return str(value) if value else fallback_stem


class Bdd100kTrackingDatasetAdapter:
    def __init__(self, config: Bdd100kTrackingDatasetConfig) -> None:
        self.config = config
        self.frame_adapter = Bdd100kDatasetAdapter(Bdd100kDatasetConfig(root=config.root, split=config.split))

    def iter_scene_windows(self, window_size: int = 4) -> Iterator[tuple[list[Bdd100kTrackingRecord], Bdd100kTrackingRecord]]:
        for video_name, samples in self._group_samples_by_video().items():
            histories = self._derive_track_histories(video_name, samples)
            for track_history in histories.values():
                ordered = sorted(track_history, key=lambda record: record.frame_index)
                if len(ordered) < window_size:
                    continue
                for end_index in range(window_size - 1, len(ordered)):
                    history = ordered[end_index - window_size + 1 : end_index]
                    target = ordered[end_index]
                    yield history, target

    def _group_samples_by_video(self) -> dict[str, list[Bdd100kFrameSample]]:
        grouped: dict[str, list[Bdd100kFrameSample]] = defaultdict(list)
        for sample in self.frame_adapter.iter_samples():
            grouped[sample.video_name].append(sample)
        for samples in grouped.values():
            samples.sort(key=lambda sample: sample.frame_index)
        return grouped

    def _derive_track_histories(self, video_name: str, samples: list[Bdd100kFrameSample]) -> dict[int, list[Bdd100kTrackingRecord]]:
        histories: dict[int, list[Bdd100kTrackingRecord]] = defaultdict(list)
        previous_records: list[Bdd100kTrackingRecord] = []
        next_track_id = 0

        for sample in samples:
            current_records: list[Bdd100kTrackingRecord] = []
            unmatched_previous = set(range(len(previous_records)))
            for annotation in sample.annotations:
                best_match_index = -1
                best_overlap = 0.0
                for previous_index in unmatched_previous:
                    previous_record = previous_records[previous_index]
                    if previous_record.object_class != annotation.object_class:
                        continue
                    overlap = bbox_iou(previous_record.bbox, annotation.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match_index = previous_index

                if best_match_index >= 0 and best_overlap >= self.config.iou_threshold:
                    track_id = previous_records[best_match_index].track_id
                    unmatched_previous.remove(best_match_index)
                else:
                    track_id = next_track_id
                    next_track_id += 1

                record = Bdd100kTrackingRecord(
                    track_id=track_id,
                    video_name=video_name,
                    frame_index=sample.frame_index,
                    object_class=annotation.object_class,
                    bbox=annotation.bbox,
                    occluded=annotation.occluded,
                    confidence=annotation.confidence,
                )
                current_records.append(record)
                histories[track_id].append(record)

            previous_records = current_records

        return histories