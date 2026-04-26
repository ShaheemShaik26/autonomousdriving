from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from ..common.vehicle_detection_dataset import VehicleDetectionDatasetAdapter, VehicleDetectionDatasetConfig
from ..common.schemas import ObjectClass, bbox_iou
from ..services.perception_service.core import PerceptionService


@dataclass(frozen=True)
class PerceptionEvaluationResult:
    frames_evaluated: int
    precision: float
    recall: float
    mean_iou: float
    classwise_recall: dict[str, float]


def evaluate_dataset(data_root: Path, checkpoint_path: Path | None = None, max_samples: int | None = None) -> PerceptionEvaluationResult:
    adapter = VehicleDetectionDatasetAdapter(VehicleDetectionDatasetConfig(data_root))
    service = PerceptionService(checkpoint_path=str(checkpoint_path) if checkpoint_path else None)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    iou_values: list[float] = []
    class_matches: dict[str, int] = {cls.value: 0 for cls in ObjectClass}
    class_totals: dict[str, int] = {cls.value: 0 for cls in ObjectClass}

    frames_evaluated = 0
    samples = adapter.iter_samples()
    if max_samples is not None:
        samples = samples[:max_samples]

    for sample in samples:
        frames_evaluated += 1
        observation = asyncio.run(service.analyze(sample.frame))
        ground_truth = adapter._annotations_to_truth(sample.annotations)
        matched_detection_indices: set[int] = set()

        for ground_truth_object in ground_truth:
            class_totals[ground_truth_object.object_class.value] += 1
            best_match = None
            best_iou = 0.0
            best_match_index = -1
            for detection_index, detection in enumerate(observation.scene.detections):
                if detection_index in matched_detection_indices:
                    continue
                if detection.object_class != ground_truth_object.object_class:
                    continue
                overlap = bbox_iou(ground_truth_object.bbox, detection.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_match = detection
                    best_match_index = detection_index
            if best_match is not None and best_iou >= 0.5:
                true_positive += 1
                iou_values.append(best_iou)
                class_matches[ground_truth_object.object_class.value] += 1
                matched_detection_indices.add(best_match_index)
            else:
                false_negative += 1

        for detection in observation.scene.detections:
            if not any(bbox_iou(detection.bbox, truth.bbox) >= 0.5 and detection.object_class == truth.object_class for truth in ground_truth):
                false_positive += 1

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    classwise_recall = {
        class_name: (class_matches[class_name] / class_totals[class_name] if class_totals[class_name] else 0.0)
        for class_name in class_totals
    }
    return PerceptionEvaluationResult(
        frames_evaluated=frames_evaluated,
        precision=precision,
        recall=recall,
        mean_iou=(sum(iou_values) / len(iou_values)) if iou_values else 0.0,
        classwise_recall=classwise_recall,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Kaggle vehicle-detection perception stack")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    result = evaluate_dataset(Path(args.data_root), Path(args.checkpoint) if args.checkpoint else None, args.max_samples)
    print(
        json.dumps(
            {
                "frames_evaluated": result.frames_evaluated,
                "precision": result.precision,
                "recall": result.recall,
                "mean_iou": result.mean_iou,
                "classwise_recall": result.classwise_recall,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
