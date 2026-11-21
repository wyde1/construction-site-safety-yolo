"""Command-line interface for Construction Site Safety PPE detection on images or videos."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

from ppe_safety.compliance import ComplianceSummary, aggregate_summaries
from ppe_safety.config import Settings
from ppe_safety.detector import SafetyDetector, YOLOConstructionDetector

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})


def _validated_conf(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid confidence value: {raw!r}") from exc
    if not math.isfinite(value) or not (0.05 <= value <= 0.95):
        raise argparse.ArgumentTypeError("Confidence threshold must be between 0.05 and 0.95.")
    return value


def _validated_max_frames(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid max-frames value: {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--max-frames must be >= 1.")
    return value


def process_image_file(
    image_path: Path,
    detector: SafetyDetector,
    output_dir: Path,
    conf: float,
    draw_overlay: bool = True,
) -> dict:
    """Run PPE detection on a single image and save the annotated output."""
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError(f"Unable to read image file: '{image_path}'")

    result = detector.predict(frame_bgr, conf=conf, draw_overlay=draw_overlay)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix or '.jpg'}"
    if not cv2.imwrite(str(out_path), result.annotated_bgr):
        raise RuntimeError(f"Failed to write annotated image to '{out_path}'.")

    return {
        "source": image_path.name,
        "output_media": out_path.name,
        "inference_ms": result.inference_ms,
        "compliance": result.compliance.to_dict(),
        "detections": [det.to_dict() for det in result.detections],
    }


def process_video_file(
    video_path: Path,
    detector: SafetyDetector,
    output_dir: Path,
    conf: float,
    draw_overlay: bool = True,
    max_frames: int | None = None,
) -> dict:
    """Run frame-by-frame PPE detection on a video and write an annotated MP4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions ({width}x{height}) in '{video_path}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to initialize video writer for '{out_path}'.")

    frame_summaries: list[ComplianceSummary] = []
    frame_records: list[dict] = []
    processed = 0

    try:
        while True:
            if max_frames is not None and processed >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            result = detector.predict(frame_bgr, conf=conf, draw_overlay=draw_overlay)
            writer.write(result.annotated_bgr)
            frame_summaries.append(result.compliance)
            frame_records.append(
                {
                    "frame_index": processed,
                    "inference_ms": result.inference_ms,
                    "has_violations": result.compliance.has_violations,
                    "violation_count": result.compliance.violation_count,
                    "total_detections": result.compliance.total_detections,
                }
            )
            processed += 1
    finally:
        cap.release()
        writer.release()

    if processed == 0:
        out_path.unlink(missing_ok=True)
        raise ValueError(f"Video file contained no readable frames: '{video_path}'")

    return {
        "source": video_path.name,
        "output_media": out_path.name,
        "fps": round(fps, 2),
        "summary": aggregate_summaries(frame_summaries),
        "frames": frame_records,
    }


def parse_args(argv: list[str] | None = None, defaults: Settings | None = None) -> argparse.Namespace:
    cfg = defaults or Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Run YOLO11 construction site PPE & machinery safety detection."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an image file, directory of images, or video file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=cfg.model_path,
        help=f"Path to trained YOLO11 .pt checkpoint (default: {cfg.model_path}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where annotated media and JSON reports will be written (default: outputs).",
    )
    parser.add_argument(
        "--conf",
        type=_validated_conf,
        default=cfg.conf_threshold,
        help=f"Confidence threshold in [0.05, 0.95] (default: {cfg.conf_threshold}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.device,
        help="Optional Ultralytics device override (e.g. 'cpu', '0').",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write a structured JSON compliance report into --output-dir.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the top-left PPE compliance summary banner on output frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=_validated_max_frames,
        default=None,
        help="Optional cap on processed frames when --source is a video file.",
    )
    return parser.parse_args(argv)


def run_cli(
    argv: list[str] | None = None,
    detector: SafetyDetector | None = None,
) -> int:
    """Execute the CLI pipeline and return a POSIX exit code."""
    args = parse_args(argv)
    source: Path = args.source

    if not source.exists():
        print(f"Error: Source path not found: '{source}'", file=sys.stderr)
        return 1

    # Validate source type and collect inputs before loading YOLO weights.
    dir_images: list[Path] = []
    if source.is_dir():
        source_mode = "directory"
        dir_images = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not dir_images:
            print(f"Error: No supported images found in '{source}'.", file=sys.stderr)
            return 1
        resolved_dir_name = source.resolve().name or "batch"
        report_prefix = resolved_dir_name
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        source_mode = "image"
        report_prefix = source.stem or "image"
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        source_mode = "video"
        report_prefix = source.stem or "video"
    else:
        print(
            f"Error: Unsupported file extension '{source.suffix}'. "
            f"Supported images: {sorted(IMAGE_EXTENSIONS)}; videos: {sorted(VIDEO_EXTENSIONS)}.",
            file=sys.stderr,
        )
        return 1

    active_detector = detector
    if active_detector is None:
        try:
            active_detector = YOLOConstructionDetector(
                model_path=args.weights,
                default_conf=args.conf,
                device=args.device,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    draw_overlay = not args.no_overlay

    try:
        if source_mode == "directory":
            items = [
                process_image_file(img, active_detector, args.output_dir, args.conf, draw_overlay)
                for img in dir_images
            ]
            summaries = [ComplianceSummary(**item["compliance"]) for item in items]
            report = {
                "mode": "directory",
                "source": report_prefix,
                "summary": aggregate_summaries(summaries),
                "items": items,
            }
            print(
                f"Processed {len(items)} image(s) -> '{args.output_dir}' | "
                f"Violations found in {report['summary']['frames_with_violations']}/{len(items)} image(s)."
            )
        elif source_mode == "image":
            item = process_image_file(source, active_detector, args.output_dir, args.conf, draw_overlay)
            report = {"mode": "image", **item}
            comp = item["compliance"]
            status = "PPE ALERT" if comp["has_violations"] else "PPE OK"
            print(
                f"[{status}] {source.name} -> '{args.output_dir / item['output_media']}' | "
                f"detections={comp['total_detections']}, violations={comp['violation_count']} "
                f"({item['inference_ms']:.1f} ms)"
            )
        else:
            video_report = process_video_file(
                source,
                active_detector,
                args.output_dir,
                args.conf,
                draw_overlay=draw_overlay,
                max_frames=args.max_frames,
            )
            report = {"mode": "video", **video_report}
            summary = video_report["summary"]
            print(
                f"Processed video '{source.name}' ({summary['frames_analyzed']} frames) -> "
                f"'{args.output_dir / video_report['output_media']}' | "
                f"frames_with_violations={summary['frames_with_violations']}"
            )

        if args.save_json:
            report_path = args.output_dir / f"{report_prefix}_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved compliance report to '{report_path}'.")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
