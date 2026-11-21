"""Pure PPE safety compliance analysis and multi-frame aggregation utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ppe_safety.config import classify_label


@dataclass(frozen=True)
class Detection:
    """Single detected construction site entity."""

    label: str
    class_id: int
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]

    @property
    def category(self) -> str:
        return classify_label(self.label)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "bbox": [round(coord, 2) for coord in self.bbox],
        }


@dataclass(frozen=True)
class ComplianceSummary:
    """Aggregated PPE safety compliance status for a single image or video frame."""

    total_detections: int
    compliant_ppe_count: int
    violation_count: int
    person_count: int
    machinery_count: int
    has_violations: bool
    counts_by_class: dict[str, int]
    counts_by_category: dict[str, int]
    violations_by_class: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "total_detections": self.total_detections,
            "compliant_ppe_count": self.compliant_ppe_count,
            "violation_count": self.violation_count,
            "person_count": self.person_count,
            "machinery_count": self.machinery_count,
            "has_violations": self.has_violations,
            "counts_by_class": dict(self.counts_by_class),
            "counts_by_category": dict(self.counts_by_category),
            "violations_by_class": dict(self.violations_by_class),
        }


def analyze_detections(detections: Sequence[Detection]) -> ComplianceSummary:
    """Compute a structured PPE compliance summary from raw detections."""
    by_class: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    violations: Counter[str] = Counter()

    for det in detections:
        by_class[det.label] += 1
        cat = det.category
        by_category[cat] += 1
        if cat == "ppe_violation":
            violations[det.label] += 1

    violation_count = sum(violations.values())
    return ComplianceSummary(
        total_detections=len(detections),
        compliant_ppe_count=by_category["ppe_compliant"],
        violation_count=violation_count,
        person_count=by_class["Person"],
        machinery_count=by_category["machinery_vehicle"],
        has_violations=violation_count > 0,
        counts_by_class=dict(sorted(by_class.items())),
        counts_by_category=dict(sorted(by_category.items())),
        violations_by_class=dict(sorted(violations.items())),
    )


def aggregate_summaries(summaries: Sequence[ComplianceSummary]) -> dict:
    """Aggregate compliance metrics across multiple images or video frames."""
    total_frames = len(summaries)
    frames_with_violations = sum(1 for s in summaries if s.has_violations)
    total_detections = sum(s.total_detections for s in summaries)
    total_violations = sum(s.violation_count for s in summaries)
    total_compliant_ppe = sum(s.compliant_ppe_count for s in summaries)

    class_totals: Counter[str] = Counter()
    violation_totals: Counter[str] = Counter()
    for s in summaries:
        class_totals.update(s.counts_by_class)
        violation_totals.update(s.violations_by_class)

    return {
        "frames_analyzed": total_frames,
        "frames_with_violations": frames_with_violations,
        "compliance_rate": round(
            (total_frames - frames_with_violations) / total_frames, 4
        ) if total_frames > 0 else 1.0,
        "total_detections": total_detections,
        "total_compliant_ppe": total_compliant_ppe,
        "total_violations": total_violations,
        "counts_by_class": dict(sorted(class_totals.items())),
        "violations_by_class": dict(sorted(violation_totals.items())),
    }
