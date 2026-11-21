"""Configuration, environment loading, and 25-class construction site safety taxonomy."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"

# Exact 25-class ordering from Roboflow Construction Site Safety v30 (CC BY 4.0)
CLASS_NAMES: tuple[str, ...] = (
    "Excavator",
    "Gloves",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "SUV",
    "Safety Cone",
    "Safety Vest",
    "bus",
    "dump truck",
    "fire hydrant",
    "machinery",
    "mini-van",
    "sedan",
    "semi",
    "trailer",
    "truck",
    "truck and trailer",
    "van",
    "vehicle",
    "wheel loader",
)

PPE_COMPLIANT_CLASSES: frozenset[str] = frozenset(
    {"Hardhat", "Safety Vest", "Mask", "Gloves"}
)
PPE_VIOLATION_CLASSES: frozenset[str] = frozenset(
    {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
)
SITE_PERSONNEL_CLASSES: frozenset[str] = frozenset(
    {"Person", "Safety Cone", "Ladder", "fire hydrant"}
)
MACHINERY_VEHICLE_CLASSES: frozenset[str] = frozenset(
    {
        "Excavator",
        "SUV",
        "bus",
        "dump truck",
        "machinery",
        "mini-van",
        "sedan",
        "semi",
        "trailer",
        "truck",
        "truck and trailer",
        "van",
        "vehicle",
        "wheel loader",
    }
)


def classify_label(label: str) -> str:
    """Map a class label to its operational safety category."""
    if label in PPE_COMPLIANT_CLASSES:
        return "ppe_compliant"
    if label in PPE_VIOLATION_CLASSES:
        return "ppe_violation"
    if label in SITE_PERSONNEL_CLASSES:
        return "site_personnel"
    if label in MACHINERY_VEHICLE_CLASSES:
        return "machinery_vehicle"
    return "other"


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Load key=value pairs from a local .env file without overriding existing env vars."""
    path = env_path or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_float(value: str | None, default: float, min_val: float, max_val: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if not math.isfinite(parsed):
        return default
    return max(min_val, min(max_val, parsed))


@dataclass(frozen=True)
class Settings:
    """Runtime settings for construction site safety inference."""

    model_path: Path = DEFAULT_MODEL_PATH
    conf_threshold: float = 0.35
    device: str | None = None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv_file(env_file)
        raw_model_path = (os.getenv("MODEL_PATH") or "").strip() or "weights/best.pt"
        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        raw_device = (os.getenv("DEVICE") or "").strip() or None
        return cls(
            model_path=model_path,
            conf_threshold=_parse_float(os.getenv("CONF_THRESHOLD"), 0.35, 0.05, 0.95),
            device=raw_device,
        )
