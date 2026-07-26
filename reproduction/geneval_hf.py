"""GenEval decision logic with a modern public Mask2Former inference backend.

The original GenEval package in this repository pins torch 2.1/mmcv 1.x, which
does not provide Blackwell kernels.  This module keeps GenEval's object/count,
color, and position rules while obtaining COCO instance masks from the public
Hugging Face conversion of Mask2Former Swin-S.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
NAME_ALIASES = {
    "keyboard": "computer keyboard",
    "mouse": "computer mouse",
    "remote": "tv remote",
    "potted plant": "potted plant",
    "dining table": "dining table",
    "tv": "tv",
}


@dataclass
class DetectedObject:
    bbox: np.ndarray
    mask: np.ndarray
    score: float


class ModernGenEval:
    def __init__(self, device: str):
        from transformers import (
            AutoImageProcessor,
            AutoProcessor,
            CLIPModel,
            Mask2FormerForUniversalSegmentation,
        )

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-coco-instance"
        )
        self.detector = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-coco-instance",
            torch_dtype=torch.float16,
        ).to(device).eval()
        self.id2label = {
            int(k): NAME_ALIASES.get(v.lower(), v.lower())
            for k, v in self.detector.config.id2label.items()
        }
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float16
        ).to(device).eval()
        self.color_cache: dict[str, torch.Tensor] = {}

    def detect(self, image) -> dict[str, list[DetectedObject]]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device, dtype=torch.float16 if value.is_floating_point() else None)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.30,
            target_sizes=[(image.height, image.width)],
            return_binary_maps=True,
        )[0]
        segmentation = processed["segmentation"]
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.detach().cpu().numpy()
        objects: dict[str, list[DetectedObject]] = defaultdict(list)
        for idx, info in enumerate(processed["segments_info"]):
            score = float(info.get("score", 1.0))
            label = self.id2label.get(int(info["label_id"]), str(info["label_id"]))
            if segmentation.ndim == 3:
                mask = segmentation[idx].astype(bool)
            else:
                mask = segmentation == int(info["id"])
            ys, xs = np.where(mask)
            if not len(xs):
                continue
            bbox = np.asarray([xs.min(), ys.min(), xs.max(), ys.max(), score], dtype=float)
            objects[label].append(DetectedObject(bbox=bbox, mask=mask, score=score))
        return dict(objects)

    @staticmethod
    def _relative_position(a: DetectedObject, b: DetectedObject) -> set[str]:
        boxes = np.array([a.bbox[:4], b.bbox[:4]]).reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised = np.maximum(np.abs(offset) - 0.1 * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised) < 1e-3) or np.linalg.norm(offset) == 0:
            return set()
        dx, dy = revised / np.linalg.norm(offset)
        relations: set[str] = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def _color(self, image, obj: DetectedObject, classname: str) -> str:
        from PIL import Image

        x1, y1, x2, y2 = [int(v) for v in obj.bbox[:4]]
        blank = Image.new("RGB", image.size, color="#999")
        masked = Image.composite(image.convert("RGB"), blank, Image.fromarray(obj.mask))
        crop = masked.crop((x1, y1, max(x1 + 1, x2 + 1), max(y1 + 1, y2 + 1)))
        if classname not in self.color_cache:
            texts = [
                template.format(color=color, classname=classname)
                for color in COLORS
                for template in (
                    "a photo of a {color} {classname}",
                    "a photo of a {color}-colored {classname}",
                    "a photo of a {color} object",
                )
            ]
            text_inputs = self.clip_processor(
                text=texts, padding=True, return_tensors="pt"
            )
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            with torch.inference_mode():
                features = self.clip_model.get_text_features(**text_inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            self.color_cache[classname] = features.reshape(len(COLORS), 3, -1).mean(1)
        image_inputs = self.clip_processor(images=crop, return_tensors="pt")
        image_inputs = {
            key: value.to(
                self.device, dtype=torch.float16 if value.is_floating_point() else None
            )
            for key, value in image_inputs.items()
        }
        with torch.inference_mode():
            feature = self.clip_model.get_image_features(**image_inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        return COLORS[int((feature @ self.color_cache[classname].T).argmax().item())]

    def evaluate(self, image, metadata: dict[str, Any]) -> dict[str, Any]:
        objects = self.detect(image)
        correct = True
        reasons: list[str] = []
        matched_groups: list[list[DetectedObject] | None] = []
        for req in metadata["include"]:
            classname = req["class"]
            found = sorted(objects.get(classname, []), key=lambda obj: obj.score, reverse=True)
            threshold = 0.9 if metadata["tag"] == "counting" else 0.3
            found = [obj for obj in found if obj.score > threshold][:16]
            expected = int(req["count"])
            matched = len(found) == expected
            if not matched:
                correct = False
                reasons.append(f"expected {expected} {classname}, found {len(found)}")
            if "color" in req and matched:
                predicted = [self._color(image, obj, classname) for obj in found]
                if predicted.count(req["color"]) != expected:
                    correct = matched = False
                    reasons.append(f"expected {expected} {req['color']} {classname}; got {predicted}")
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                target = matched_groups[target_group]
                if target is None:
                    correct = matched = False
                    reasons.append(f"missing target group for {classname} {expected_rel}")
                else:
                    for obj in found:
                        for target_obj in target:
                            if expected_rel not in self._relative_position(obj, target_obj):
                                correct = matched = False
                                reasons.append(f"expected {classname} {expected_rel} target")
                                break
                        if not matched:
                            break
            matched_groups.append(found if matched else None)
        for req in metadata.get("exclude", []):
            if len(objects.get(req["class"], [])) >= int(req["count"]):
                correct = False
                reasons.append(f"excluded class present: {req['class']}")
        return {
            "correct": bool(correct),
            "tag": metadata["tag"],
            "reason": "; ".join(reasons),
            "detected_classes": sorted(objects),
        }
