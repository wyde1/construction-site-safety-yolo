"""Unit tests for 25-class taxonomy and PPE compliance analysis."""

from __future__ import annotations

from ppe_safety.compliance import Detection, aggregate_summaries, analyze_detections
from ppe_safety.config import (
    CLASS_NAMES,
    MACHINERY_VEHICLE_CLASSES,
    PPE_COMPLIANT_CLASSES,
    PPE_VIOLATION_CLASSES,
    SITE_PERSONNEL_CLASSES,
    classify_label,
)


def test_taxonomy_covers_all_25_classes_disjointly() -> None:
    assert len(CLASS_NAMES) == 25
    groups = [
        PPE_COMPLIANT_CLASSES,
        PPE_VIOLATION_CLASSES,
        SITE_PERSONNEL_CLASSES,
        MACHINERY_VEHICLE_CLASSES,
    ]
    union = set().union(*groups)
    assert union == set(CLASS_NAMES)
    assert sum(len(g) for g in groups) == len(CLASS_NAMES)
    assert classify_label("UnknownClass") == "other"


def test_analyze_compliant_scene() -> None:
    detections = [
        Detection(label="Person", class_id=8, confidence=0.92, bbox=[10, 20, 100, 200]),
        Detection(label="Hardhat", class_id=2, confidence=0.89, bbox=[30, 20, 75, 55]),
        Detection(label="Safety Vest", class_id=11, confidence=0.86, bbox=[25, 70, 90, 150]),
        Detection(label="Excavator", class_id=0, confidence=0.81, bbox=[200, 100, 400, 300]),
    ]
    summary = analyze_detections(detections)

    assert summary.has_violations is False
    assert summary.total_detections == 4
    assert summary.compliant_ppe_count == 2
    assert summary.violation_count == 0
    assert summary.person_count == 1
    assert summary.machinery_count == 1
    assert summary.violations_by_class == {}


def test_analyze_scene_with_ppe_violations() -> None:
    detections = [
        Detection(label="Person", class_id=8, confidence=0.90, bbox=[10, 20, 100, 200]),
        Detection(label="NO-Hardhat", class_id=5, confidence=0.84, bbox=[30, 20, 75, 55]),
        Detection(label="NO-Safety Vest", class_id=7, confidence=0.79, bbox=[25, 70, 90, 150]),
        Detection(label="NO-Hardhat", class_id=5, confidence=0.72, bbox=[120, 20, 160, 55]),
    ]
    summary = analyze_detections(detections)

    assert summary.has_violations is True
    assert summary.violation_count == 3
    assert summary.violations_by_class == {"NO-Hardhat": 2, "NO-Safety Vest": 1}


def test_aggregate_summaries() -> None:
    ok_summary = analyze_detections(
        [Detection(label="Hardhat", class_id=2, confidence=0.9, bbox=[0, 0, 10, 10])]
    )
    alert_summary = analyze_detections(
        [Detection(label="NO-Hardhat", class_id=5, confidence=0.8, bbox=[0, 0, 10, 10])]
    )
    agg = aggregate_summaries([ok_summary, alert_summary, ok_summary])

    assert agg["frames_analyzed"] == 3
    assert agg["frames_with_violations"] == 1
    assert agg["compliance_rate"] == 0.6667
    assert agg["total_compliant_ppe"] == 2
    assert agg["total_violations"] == 1


def test_aggregate_summaries_empty() -> None:
    agg = aggregate_summaries([])
    assert agg["frames_analyzed"] == 0
    assert agg["frames_with_violations"] == 0
    assert agg["compliance_rate"] == 1.0
