from __future__ import annotations

from dataclasses import dataclass
import random

from .schemas import BoundingBox, ObjectClass


@dataclass(frozen=True)
class SyntheticTrackRecord:
    track_id: int
    frame_index: int
    object_class: ObjectClass
    bbox: BoundingBox
    velocity_x: float
    velocity_y: float
    occluded: int = 0


@dataclass(frozen=True)
class SyntheticTrackingDatasetConfig:
    sequences: int = 96
    history_window: int = 4
    seed: int = 17


class SyntheticTrackingDataset:
    def __init__(self, config: SyntheticTrackingDatasetConfig | None = None) -> None:
        self.config = config or SyntheticTrackingDatasetConfig()
        self._rng = random.Random(self.config.seed)

    def iter_scene_windows(self, window_size: int | None = None):
        window_size = window_size or self.config.history_window
        for sequence_index in range(self.config.sequences):
            yield from self._build_sequence(sequence_index, window_size)

    def _build_sequence(self, sequence_index: int, window_size: int):
        object_count = self._rng.randint(2, 6)
        tracks: dict[int, list[SyntheticTrackRecord]] = {}
        for track_id in range(object_count):
            object_class = self._rng.choices(
                [ObjectClass.vehicle, ObjectClass.pedestrian, ObjectClass.obstacle],
                weights=[0.7, 0.2, 0.1],
                k=1,
            )[0]
            start_x = self._rng.uniform(40.0, 400.0)
            start_y = self._rng.uniform(60.0, 220.0)
            width = self._rng.uniform(30.0, 120.0)
            height = self._rng.uniform(25.0, 100.0)
            velocity_x = self._rng.uniform(-6.0, 8.0)
            velocity_y = self._rng.uniform(-1.0, 4.0)
            history: list[SyntheticTrackRecord] = []
            for frame_index in range(window_size):
                x1 = start_x + velocity_x * frame_index
                y1 = start_y + velocity_y * frame_index
                record = SyntheticTrackRecord(
                    track_id=track_id,
                    frame_index=frame_index,
                    object_class=object_class,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height),
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    occluded=1 if frame_index == window_size - 1 and track_id % 3 == 0 else 0,
                )
                history.append(record)
            tracks[track_id] = history

        for track_history in tracks.values():
            if len(track_history) < window_size:
                continue
            yield track_history[:-1], track_history[-1]