from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ...common.schemas import BoundingBox, FrameEnvelope, ObjectClass, ObjectState, SceneObservation, bbox_iou, clamp_probability
from ...common.simulation import decode_image

try:
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.transforms.functional import to_tensor
except Exception:  # pragma: no cover - fallback when torchvision is unavailable
    torchvision = None
    FasterRCNN_ResNet50_FPN_Weights = None
    to_tensor = None


class TinyObjectClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 3))
        linear = self.net[-1]
        assert isinstance(linear, nn.Linear)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [
                        [2.8, -0.7, -0.7],
                        [-0.7, 2.8, -0.7],
                        [-0.7, -0.7, 2.8],
                    ],
                    dtype=torch.float32,
                )
            )
            linear.bias.copy_(torch.tensor([-0.2, -0.2, -0.2], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PerceptionResult:
    scene: SceneObservation
    model_latency_ms: float


class PerceptionService:
    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.device = torch.device("cpu")
        self.detection_score_threshold = float(os.getenv("PERCEPTION_SCORE_THRESHOLD", "0.3"))
        self.detector = self._build_detector(checkpoint_path)
        self.model = TinyObjectClassifier()
        self.model.eval()
        if checkpoint_path is not None and checkpoint_path.endswith(".pt"):
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "classifier" in state:
                self.model.load_state_dict(state["classifier"])

    def _build_detector(self, checkpoint_path: str | None) -> torch.nn.Module | None:
        if torchvision is None:
            return None
        if checkpoint_path is not None and checkpoint_path.endswith(".pth"):
            weights = torch.load(checkpoint_path, map_location="cpu")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            model.load_state_dict(weights)
            model.eval()
            return model
        if FasterRCNN_ResNet50_FPN_Weights is not None:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        model.eval()
        return model

    async def analyze(self, frame: FrameEnvelope) -> PerceptionResult:
        start = asyncio.get_running_loop().time()
        image = decode_image(frame.encoded_image)
        detections = self._detect_objects(image)
        if not detections:
            proposals = self._find_regions(image)
            detections = [self._classify_region(image, proposal, index) for index, proposal in enumerate(proposals)]
            detections = self._merge_overlapping_detections(detections)
        density = self._estimate_density(detections, frame.width, frame.height)
        latency_ms = (asyncio.get_running_loop().time() - start) * 1000.0
        observation = SceneObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            detections=detections,
            object_density=density,
            camera_health="ok",
        )
        return PerceptionResult(scene=observation, model_latency_ms=latency_ms)

    def _detect_objects(self, image: np.ndarray) -> list[ObjectState]:
        if self.detector is None or to_tensor is None:
            return []
        tensor = to_tensor(image.copy()).to(self.device)
        with torch.no_grad():
            outputs = self.detector([tensor])[0]

        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        boxes = outputs["boxes"].detach().cpu().numpy()

        detections: list[ObjectState] = []
        for index, (score, label, box) in enumerate(zip(scores, labels, boxes, strict=False)):
            if score < self.detection_score_threshold:
                continue
            object_class = self._map_coco_label(int(label))
            if object_class is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(
                ObjectState(
                    object_id=f"det_{index}",
                    object_class=object_class,
                    confidence=clamp_probability(float(score)),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    velocity_x=0.0,
                    velocity_y=0.0,
                    track_age=1,
                )
            )
        return self._merge_overlapping_detections(detections)

    def _map_coco_label(self, label: int) -> ObjectClass | None:
        if label in {3, 6, 8}:
            return ObjectClass.vehicle
        return None

    def _find_regions(self, image: np.ndarray) -> list[BoundingBox]:
        proposals: list[BoundingBox] = []
        proposals.extend(self._connected_components(self._color_mask(image, class_name="vehicle")))
        expanded = [self._expand_bbox(proposal, image.shape[1], image.shape[0], scale=1.12) for proposal in proposals]
        return self._non_max_suppression(expanded)

    def _color_mask(self, image: np.ndarray, class_name: str) -> np.ndarray:
        pixels = image.astype(np.int16)
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        if class_name == "vehicle":
            return (red > green + 35) & (red > blue + 35) & (red > 90)
        return np.zeros_like(red, dtype=bool)

    def _connected_components(self, mask: np.ndarray) -> list[BoundingBox]:
        if not mask.any():
            return []
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        proposals: list[BoundingBox] = []
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for row in range(height):
            for col in range(width):
                if not mask[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                min_row = max_row = row
                min_col = max_col = col
                component_size = 0
                while stack:
                    current_row, current_col = stack.pop()
                    component_size += 1
                    min_row = min(min_row, current_row)
                    max_row = max(max_row, current_row)
                    min_col = min(min_col, current_col)
                    max_col = max(max_col, current_col)
                    for delta_row, delta_col in neighbors:
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
                if component_size >= 60:
                    proposals.append(
                        BoundingBox(
                            x1=float(min_col),
                            y1=float(min_row),
                            x2=float(max_col + 1),
                            y2=float(max_row + 1),
                        )
                    )
        return proposals

    def _classify_region(self, image: np.ndarray, bbox: BoundingBox, index: int) -> ObjectState:
        crop = self._crop(image, bbox)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
        if tensor.shape[-1] < 8 or tensor.shape[-2] < 8:
            tensor = torch.nn.functional.interpolate(tensor, size=(32, 32), mode="bilinear", align_corners=False)
        logits = self.model(tensor)
        confidence = float(torch.softmax(logits, dim=1)[0, 0].item())
        center_x, center_y = bbox.center
        velocity_x = float((center_x - image.shape[1] / 2.0) / max(image.shape[1], 1) * 12.0)
        velocity_y = float((center_y - image.shape[0] / 2.0) / max(image.shape[0], 1) * 4.0)
        return ObjectState(
            object_id=f"det_{index}",
            object_class=ObjectClass.vehicle,
            confidence=clamp_probability(confidence),
            bbox=bbox,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            track_age=1,
        )

    def _estimate_density(self, detections: list[ObjectState], width: int, height: int) -> float:
        occupied = sum(d.bbox.area for d in detections)
        return min(1.0, occupied / max(float(width * height), 1.0))

    def _crop(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        x1 = max(0, int(math.floor(bbox.x1)))
        y1 = max(0, int(math.floor(bbox.y1)))
        x2 = min(image.shape[1], max(x1 + 1, int(math.ceil(bbox.x2))))
        y2 = min(image.shape[0], max(y1 + 1, int(math.ceil(bbox.y2))))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image[:32, :32]
        return crop.astype(np.float32) / 255.0

    def _non_max_suppression(self, proposals: list[BoundingBox], threshold: float = 0.4) -> list[BoundingBox]:
        if not proposals:
            return []
        selected: list[BoundingBox] = []
        for proposal in sorted(proposals, key=lambda item: item.area, reverse=True):
            if all(bbox_iou(proposal, existing) < threshold for existing in selected):
                selected.append(proposal)
        return selected

    def _merge_overlapping_detections(self, detections: list[ObjectState]) -> list[ObjectState]:
        merged: list[ObjectState] = []
        for detection in detections:
            if all(bbox_iou(detection.bbox, other.bbox) < 0.6 for other in merged):
                merged.append(detection)
        return merged

    def _expand_bbox(self, bbox: BoundingBox, width: int, height: int, scale: float = 1.0) -> BoundingBox:
        center_x, center_y = bbox.center
        half_width = bbox.width * scale / 2.0
        half_height = bbox.height * scale / 2.0
        return BoundingBox(
            x1=max(0.0, center_x - half_width),
            y1=max(0.0, center_y - half_height),
            x2=min(float(width), center_x + half_width),
            y2=min(float(height), center_y + half_height),
        )
