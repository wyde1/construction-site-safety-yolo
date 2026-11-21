"""CLI, detector, config, and weight-downloader tests.

Default tests use a mock detector and pass even when ``weights/best.pt`` is absent.
Tests marked ``@pytest.mark.real_model`` exercise the actual YOLO11s checkpoint when available.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ppe_safety.cli import run_cli
from ppe_safety.compliance import Detection, analyze_detections
from ppe_safety.config import CLASS_NAMES, CONFIGS_DIR, PROJECT_ROOT, Settings
from ppe_safety.detector import InferenceResult, YOLOConstructionDetector, draw_detections
from scripts.download_weights import download_weights


class MockSafetyDetector:
    """Deterministic weight-free detector for fast CLI and pipeline testing."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._names = {idx: name for idx, name in enumerate(CLASS_NAMES)}
        self._detections = detections if detections is not None else [
            Detection(label="Person", class_id=8, confidence=0.91, bbox=[20.0, 20.0, 120.0, 210.0]),
            Detection(label="Hardhat", class_id=2, confidence=0.88, bbox=[40.0, 22.0, 95.0, 58.0]),
            Detection(label="NO-Safety Vest", class_id=7, confidence=0.65, bbox=[30.0, 75.0, 110.0, 160.0]),
        ]
        self.last_conf: float | None = None

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: float | None = None,
        draw_overlay: bool = True,
    ) -> InferenceResult:
        self.last_conf = conf
        threshold = 0.35 if conf is None else conf
        filtered = [d for d in self._detections if d.confidence >= threshold]
        compliance = analyze_detections(filtered)
        annotated = draw_detections(frame_bgr, filtered, compliance, draw_banner=draw_overlay)
        return InferenceResult(
            annotated_bgr=annotated,
            detections=filtered,
            compliance=compliance,
            inference_ms=5.4,
        )


def _write_sample_image(path: Path, width: int = 320, height: int = 240) -> Path:
    img = np.full((height, width, 3), 135, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


def _write_sample_video(path: Path, frames: int = 4, width: int = 320, height: int = 240) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            img = np.full((height, width, 3), 80 + idx * 20, dtype=np.uint8)
            writer.write(img)
    finally:
        writer.release()
    return path


def test_yaml_configs_are_valid_and_portable() -> None:
    augment_cfg = yaml.safe_load((CONFIGS_DIR / "train_augment.yaml").read_text(encoding="utf-8"))
    assert augment_cfg["pretrained"] is True
    assert augment_cfg["flipud"] == 0.0
    assert "data" not in augment_cfg

    data_cfg = yaml.safe_load((CONFIGS_DIR / "data.example.yaml").read_text(encoding="utf-8"))
    assert data_cfg["nc"] == 25
    assert tuple(data_cfg["names"]) == CLASS_NAMES
    assert data_cfg["roboflow"]["license"] == "CC BY 4.0"


def test_cli_single_image_and_json_report(tmp_path: Path) -> None:
    img_path = _write_sample_image(tmp_path / "site_01.jpg")
    out_dir = tmp_path / "out"
    detector = MockSafetyDetector()

    exit_code = run_cli(
        ["--source", str(img_path), "--output-dir", str(out_dir), "--conf", "0.50", "--save-json"],
        detector=detector,
    )
    assert exit_code == 0
    assert detector.last_conf == pytest.approx(0.50)
    assert (out_dir / "site_01_annotated.jpg").is_file()

    report = json.loads((out_dir / "site_01_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "image"
    assert report["compliance"]["has_violations"] is True
    assert report["compliance"]["violation_count"] == 1
    assert report["compliance"]["compliant_ppe_count"] == 1


def test_cli_directory_batch_and_threshold_filtering(tmp_path: Path) -> None:
    src_dir = tmp_path / "batch"
    src_dir.mkdir()
    _write_sample_image(src_dir / "a.jpg")
    _write_sample_image(src_dir / "b.png")
    out_dir = tmp_path / "batch_out"

    # At conf=0.80 the 0.65 NO-Safety Vest detection is filtered out -> 0 violations.
    exit_code = run_cli(
        ["--source", str(src_dir), "--output-dir", str(out_dir), "--conf", "0.80", "--save-json"],
        detector=MockSafetyDetector(),
    )
    assert exit_code == 0
    report = json.loads((out_dir / "batch_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "directory"
    assert report["summary"]["frames_analyzed"] == 2
    assert report["summary"]["frames_with_violations"] == 0
    assert report["summary"]["compliance_rate"] == 1.0


def test_cli_video_processing_with_max_frames(tmp_path: Path) -> None:
    video_path = _write_sample_video(tmp_path / "clip.mp4", frames=5)
    out_dir = tmp_path / "video_out"

    exit_code = run_cli(
        [
            "--source",
            str(video_path),
            "--output-dir",
            str(out_dir),
            "--conf",
            "0.35",
            "--max-frames",
            "3",
            "--no-overlay",
            "--save-json",
        ],
        detector=MockSafetyDetector(),
    )
    assert exit_code == 0
    assert (out_dir / "clip_annotated.mp4").is_file()

    report = json.loads((out_dir / "clip_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "video"
    assert report["summary"]["frames_analyzed"] == 3
    assert report["summary"]["frames_with_violations"] == 3


def test_cli_dot_directory_report_naming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site_dir = tmp_path / "site_v1.0"
    site_dir.mkdir()
    _write_sample_image(site_dir / "frame1.jpg")
    out_dir = tmp_path / "dot_out"

    monkeypatch.chdir(site_dir)
    exit_code = run_cli(
        ["--source", ".", "--output-dir", str(out_dir), "--conf", "0.35", "--save-json"],
        detector=MockSafetyDetector(),
    )
    assert exit_code == 0
    report_path = out_dir / "site_v1.0_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source"] == "site_v1.0"


def test_cli_error_paths(tmp_path: Path) -> None:
    # Missing source
    assert run_cli(["--source", str(tmp_path / "missing.jpg")], detector=MockSafetyDetector()) == 1

    # Unsupported extension (rejected before loading weights even without mock)
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello", encoding="utf-8")
    assert run_cli(["--source", str(txt_file), "--weights", str(tmp_path / "no_weights.pt")]) == 1

    # Empty directory (rejected before loading weights)
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert run_cli(["--source", str(empty_dir), "--weights", str(tmp_path / "no_weights.pt")]) == 1

    # Missing weights without injected mock on valid image
    img_path = _write_sample_image(tmp_path / "img.jpg")
    assert run_cli(["--source", str(img_path), "--weights", str(tmp_path / "no_weights.pt")]) == 1


def test_settings_loads_dotenv_without_overriding_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CONF_THRESHOLD=0.60\nMODEL_PATH=custom/model.pt\n", encoding="utf-8")
    # Set then delete so monkeypatch always restores absence of CONF_THRESHOLD on teardown.
    monkeypatch.setenv("CONF_THRESHOLD", "")
    monkeypatch.delenv("CONF_THRESHOLD")
    monkeypatch.setenv("MODEL_PATH", "override.pt")

    cfg = Settings.from_env(env_file=env_file)
    assert cfg.conf_threshold == pytest.approx(0.60)
    assert cfg.model_path.name == "override.pt"


def test_download_weights_404_gives_manual_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise_404(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="https://example.invalid/best.pt",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_404)
    target = tmp_path / "weights" / "best.pt"
    with pytest.raises(RuntimeError) as exc_info:
        download_weights("https://example.invalid/best.pt", output_path=target)

    msg = str(exc_info.value)
    assert "404" in msg
    assert str(target) in msg
    assert "MODEL_PATH" in msg
    assert not target.with_suffix(".pt.part").exists()


def test_download_weights_cleans_up_part_on_midstream_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _BrokenResponse:
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int = -1) -> bytes:
            raise TimeoutError("connection reset mid-stream")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_kw: _BrokenResponse())
    target = tmp_path / "weights" / "best.pt"
    with pytest.raises(RuntimeError, match="Network/IO error"):
        download_weights("https://example.invalid/best.pt", output_path=target)
    assert not target.exists()
    assert not target.with_suffix(".pt.part").exists()


# --- Real model integration tests (skipped automatically if weights/best.pt is absent) ---

REAL_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best.pt"


@pytest.mark.real_model
@pytest.mark.skipif(not REAL_WEIGHTS_PATH.is_file(), reason="weights/best.pt not present locally")
def test_real_yolo_construction_model_inference(tmp_path: Path) -> None:
    detector = YOLOConstructionDetector(model_path=REAL_WEIGHTS_PATH, default_conf=0.35)
    assert len(detector.class_names) == 25
    assert set(detector.class_names.values()) == set(CLASS_NAMES)

    sample = _write_sample_image(tmp_path / "blank.jpg", width=640, height=480)
    out_dir = tmp_path / "real_out"
    exit_code = run_cli(
        ["--source", str(sample), "--output-dir", str(out_dir), "--conf", "0.35", "--save-json"],
        detector=detector,
    )
    assert exit_code == 0
    report = json.loads((out_dir / "blank_report.json").read_text(encoding="utf-8"))
    assert report["inference_ms"] > 0
    assert isinstance(report["detections"], list)
