"""Train a YOLO fire-detection model from the local dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402  (config path must be set first)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a YOLO model using data.yaml."
    )
    parser.add_argument("--data", default="data.yaml", help="Path to YOLO data YAML.")
    parser.add_argument(
        "--model",
        default="yolo11n.yaml",
        help=(
            "YOLO model config or weights. Use yolo11n.yaml to train from scratch, "
            "or yolo11n.pt if you have/download pretrained weights."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--device", default=None, help="Device, e.g. cpu, 0, or 0,1.")
    parser.add_argument(
        "--project", default="runs/detect", help="Directory for training outputs."
    )
    parser.add_argument("--name", default="fire_train", help="Run name.")
    parser.add_argument(
        "--copy-best",
        default="best.pt",
        help="Copy final best weights here for app.py. Use empty string to skip.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(resolve_path(args.project)),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if args.copy_best:
        destination = resolve_path(args.copy_best)
        if best_weights.is_file():
            shutil.copy2(best_weights, destination)
            print(f"Copied best weights to {destination}")
        else:
            print(f"Training finished, but best weights were not found at {best_weights}")

    if best_weights.is_file():
        YOLO(str(best_weights)).val(data=str(data_path), imgsz=args.imgsz, split="test")


if __name__ == "__main__":
    main()
