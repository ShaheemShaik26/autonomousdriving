from __future__ import annotations

import base64
import io
import random
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .schemas import BoundingBox, FrameEnvelope, GroundTruthObject, ObjectClass, SyntheticScene


_CLASS_COLORS = {
    ObjectClass.vehicle: (220, 70, 70),
    ObjectClass.pedestrian: (70, 210, 90),
    ObjectClass.obstacle: (70, 120, 220),
}


@dataclass(frozen=True)
class SyntheticFrameConfig:
    width: int = 960
    height: int = 540
    max_objects: int = 10
    seed: int = 7


class SyntheticSceneGenerator:
    def __init__(self, config: SyntheticFrameConfig | None = None) -> None:
        self.config = config or SyntheticFrameConfig()
        self._rng = random.Random(self.config.seed)

    def generate(self, sequence_id: int) -> SyntheticScene:
        width = self.config.width
        height = self.config.height
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background[:, :] = np.array([18, 20, 26], dtype=np.uint8)
        image = Image.fromarray(background, mode="RGB")
        draw = ImageDraw.Draw(image)

        lane_y = height * 0.65
        draw.line((0, lane_y, width, lane_y), fill=(100, 100, 100), width=4)
        draw.rectangle((width * 0.45, 0, width * 0.55, height), outline=(45, 45, 45), width=1)

        objects: list[GroundTruthObject] = []
        object_count = self._rng.randint(4, self.config.max_objects)
        timestamp_ms = int(time.time() * 1000) + sequence_id * 33

        for index in range(object_count):
            obj_class = self._rng.choices(
                [ObjectClass.vehicle, ObjectClass.pedestrian, ObjectClass.obstacle],
                weights=[0.6, 0.25, 0.15],
                k=1,
            )[0]
            bbox = self._sample_bbox(obj_class, width, height, index)
            velocity_x = self._rng.uniform(-12.0, 12.0)
            velocity_y = self._rng.uniform(-1.5, 4.5)
            color = _CLASS_COLORS[obj_class]
            draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), fill=color, outline=(255, 255, 255), width=2)
            objects.append(
                GroundTruthObject(
                    object_id=f"obj_{sequence_id}_{index}",
                    object_class=obj_class,
                    bbox=bbox,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                )
            )

        encoded_image = _encode_image(image)
        frame = FrameEnvelope(
            frame_id=f"frame_{sequence_id}",
            timestamp_ms=timestamp_ms,
            sequence_id=sequence_id,
            width=width,
            height=height,
            encoded_image=encoded_image,
            source="synthetic",
        )
        return SyntheticScene(frame=frame, objects=objects)

    def _sample_bbox(self, obj_class: ObjectClass, width: int, height: int, index: int) -> BoundingBox:
        class_scale = {
            ObjectClass.vehicle: (70, 130),
            ObjectClass.pedestrian: (25, 45),
            ObjectClass.obstacle: (35, 80),
        }[obj_class]
        box_width = self._rng.randint(*class_scale)
        box_height = self._rng.randint(int(class_scale[0] * 0.8), int(class_scale[1] * 1.2))
        x1 = self._rng.randint(20, max(21, width - box_width - 20))
        y1 = self._rng.randint(int(height * 0.25), max(int(height * 0.25) + 1, height - box_height - 20))
        if index % 3 == 0:
            y1 = min(y1, int(height * 0.55))
        return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x1 + box_width), y2=float(y1 + box_height))


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(encoded_image: str) -> np.ndarray:
    raw = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def preprocess_frame(encoded_image: str) -> np.ndarray:
    array = decode_image(encoded_image)
    return array.astype(np.float32) / 255.0


def estimate_object_density(objects: list[GroundTruthObject], width: int, height: int) -> float:
    image_area = float(width * height)
    occupied_area = sum(obj.bbox.area for obj in objects)
    return min(1.0, occupied_area / max(image_area, 1.0))


def build_scene(frame: FrameEnvelope, objects: list[GroundTruthObject]) -> SyntheticScene:
    return SyntheticScene(frame=frame, objects=objects)
