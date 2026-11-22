"""Train YOLO11n-pose on the IPN hand pose dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "pose_smoke" / "pose.yaml")
    p.add_argument("--model", type=Path,
                   default=ROOT / "weights" / "yolo11n-pose.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--project", type=Path, default=ROOT / "runs" / "pose")
    p.add_argument("--name", default="smoke")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--predict", type=int, default=6,
                   help="after training, overlay predictions on N val images")
    args = p.parse_args()

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        plots=True,
        verbose=True,
    )

    best = args.project / args.name / "weights" / "best.pt"
    if best.exists():
        model = YOLO(str(best))
    metrics = model.val(data=str(args.data))
    print(metrics)

    if args.predict:
        val_list = args.data.parent / "val.txt"
        lines = val_list.read_text(encoding="utf-8").splitlines()[: args.predict]
        images = [line if Path(line).is_absolute() else str(ROOT / line) for line in lines]
        model.predict(
            source=images,
            save=True,
            conf=0.25,
            project=str(args.project),
            name=f"{args.name}_pred",
            exist_ok=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
