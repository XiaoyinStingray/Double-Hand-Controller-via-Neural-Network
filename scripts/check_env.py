"""Verify that the runtime environment is ready.

Usage:
    python scripts/check_env.py            # core checks (works headless)
    python scripts/check_env.py --require-app   # also require desktop/demo deps
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

CORE_MODULES = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "ultralytics",
    "mediapipe",
    "onnx",
    "onnxruntime",
    "sklearn",
    "pandas",
    "matplotlib",
    "seaborn",
    "supervision",
    "yaml",
]

APP_MODULES = [
    "pyautogui",
    "fastapi",
    "uvicorn",
    "websockets",
]

REQUIRED_FILES = [
    "yolo11n.pt",
    "hand_landmarker.task",
]


def check_modules(names, errors, required=True):
    for name in names:
        try:
            mod = importlib.import_module(name)
            print(f"{name:14s}{getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:14s}{'MISSING' if required else 'not installed'} ({exc})")
            if required:
                errors.append(f"module {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-app", action="store_true",
                        help="fail when desktop/app deps (pyautogui, fastapi, ...) are missing")
    args = parser.parse_args()

    errors = []
    print(f"python        {sys.version.split()[0]}\n")

    check_modules(CORE_MODULES, errors, required=True)
    print()
    check_modules(APP_MODULES, errors, required=args.require_app)

    try:
        import torch

        print(f"\ntorch cuda    available={torch.cuda.is_available()} "
              f"version={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"gpu           {props.name} ({props.total_memory / 2**30:.1f} GB)")
            x = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float16)
            y = torch.nn.Conv2d(3, 8, 3, padding=1).cuda().half()(x)
            print(f"fp16 fwd      OK {tuple(y.shape)}")
        else:
            errors.append("torch.cuda.is_available() == False")
    except Exception as exc:  # noqa: BLE001
        print(f"torch check failed: {exc}")
        errors.append("torch cuda")

    try:
        import onnxruntime as ort

        print(f"ort providers {ort.get_available_providers()}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"onnxruntime ({exc})")

    try:
        from mediapipe.tasks.python import vision  # noqa: F401

        print("mp tasks      OK")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mediapipe tasks ({exc})")

    print()
    for name in REQUIRED_FILES:
        path = WEIGHTS / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[ok]   weights/{name} ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"[miss] weights/{name}  (run: python scripts/download_models.py)")
            errors.append(f"weights/{name}")

    if errors:
        print("\nFAILED: " + ", ".join(errors))
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
