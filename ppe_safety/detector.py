"""YOLO11 construction site safety detector and compliance banner annotation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from ppe_safety.compliance import ComplianceSummary, Detection, analyze_detections

# Distinct high-contrast BGR palette by operational category
CATEGORY_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "ppe_compliant": (80, 175, 45),      # Emerald green
    "ppe_violation": (55, 55, 225),      # Alert crimson
    "site_personnel": (235, 130, 30),    # Cobalt blue
    "machinery_vehicle": (30, 175, 245), # Amber
    "other": (160, 160, 160),            # Neutral gray
}


@dataclass(frozen=True)
class InferenceResult:
    """Output of running construction site safety detection on a single BGR frame."""

    annotated_bgr: np.ndarray
    detections: list[Detection]
    compliance: ComplianceSummary
    inference_ms: float


@runtime_checkable
class SafetyDetector(Protocol):
    """Protocol implemented by YOLOConstructionDetector and test mocks."""

    @property
    def class_names(self) -> dict[int, str]: ...

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: float | None = None,
        draw_overlay: bool = True,
    ) -> InferenceResult: ...


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    compliance: ComplianceSummary,
    draw_banner: bool = True,
) -> np.ndarray:
    """Render category-colored bounding boxes and an optional compliance HUD banner."""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, int(round(min(height, width) / 420)))
    font_scale = max(0.38, min(0.65, min(height, width) / 1100.0))

    for det in detections:
        x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        color = CATEGORY_COLORS_BGR.get(det.category, CATEGORY_COLORS_BGR["other"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(th + baseline + 5, y1)
        right = min(width - 1, x1 + tw + 6)
        cv2.rectangle(canvas, (x1, top - th - baseline - 5), (right, top), color, cv2.FILLED)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if draw_banner:
        if compliance.has_violations:
            viol_summary = ", ".join(
                f"{count} {name}" for name, count in compliance.violations_by_class.items()
            )
            banner_text = f"PPE ALERT | {viol_summary}"
            bg_color = (42, 42, 195)
        else:
            banner_text = (
                f"PPE OK | Compliant: {compliance.compliant_ppe_count} | "
                f"Persons: {compliance.person_count} | Machinery: {compliance.machinery_count}"
            )
            bg_color = (60, 138, 40)

        banner_scale = max(0.42, min(0.65, width / 1200.0))
        (bw, bh), b_base = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, 1)
        pad_x, pad_y = 10, 7
        box_w = min(width - 16, bw + pad_x * 2)
        box_h = bh + b_base + pad_y * 2
        cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), bg_color, cv2.FILLED)
        cv2.putText(
            canvas,
            banner_text,
            (8 + pad_x, 8 + pad_y + bh),
            cv2.FONT_HERSHEY_SIMPLEX,
            banner_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


class YOLOConstructionDetector:
    """Wrapper around Ultralytics YOLO11 for 25-class construction site safety detection."""

    def __init__(
        self,
        model_path: Path,
        default_conf: float = 0.35,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.default_conf = max(0.05, min(0.95, float(default_conf)))
        self.device = device
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'. "
                "Run 'python scripts/download_weights.py' or place 'best.pt' in the weights/ directory."
            )

        # Lazy import keeps unit tests with mocked detectors fast and lightweight.
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        self._predict_lock = threading.Lock()

    @property
    def class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: float | None = None,
        draw_overlay: bool = True,
    ) -> InferenceResult:
        """Run inference on a single BGR frame and return detections + compliance summary."""
        threshold = self.default_conf if conf is None else max(0.05, min(0.95, float(conf)))

        with self._predict_lock:
            names = self.class_names
            start = time.perf_counter()
            kwargs: dict = {"conf": threshold, "verbose": False}
            if self.device is not None:
                kwargs["device"] = self.device
            results = self._model(frame_bgr, **kwargs)
            result = results[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy_list = result.boxes.xyxy.cpu().tolist()
            conf_list = result.boxes.conf.cpu().tolist()
            cls_list = result.boxes.cls.cpu().tolist()
            for coords, score, cls_raw in zip(xyxy_list, conf_list, cls_list):
                cls_id = int(cls_raw)
                detections.append(
                    Detection(
                        label=names.get(cls_id, str(cls_id)),
                        class_id=cls_id,
                        confidence=float(score),
                        bbox=[float(c) for c in coords],
                    )
                )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        compliance = analyze_detections(detections)
        annotated = draw_detections(frame_bgr, detections, compliance, draw_banner=draw_overlay)
        return InferenceResult(
            annotated_bgr=annotated,
            detections=detections,
            compliance=compliance,
            inference_ms=round(elapsed_ms, 2),
        )
