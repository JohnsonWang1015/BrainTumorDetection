"""Train YOLOv8 brain-tumor detection model on Ultralytics brain-tumor dataset."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 brain tumor detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/Ultralytics/brain-tumor-abs.yaml"),
        help="Path to dataset YAML",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model variant")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to copy best weights to",
    )
    parser.add_argument("--name", default="yolo_brain_tumor", help="Run name")
    parser.add_argument("--project", default="outputs/yolo", help="YOLO project dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(args.model)

    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=0,
        amp=True,
        cos_lr=True,
        plots=True,
    )

    # Copy best weights to checkpoints/
    # YOLO prepends runs/detect/ to the project path
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path("runs/detect") / args.project / args.name / "weights" / "best.pt"
    if best_weights.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dest = args.output_dir / "yolov8_brain_tumor_best.pt"
        import shutil
        shutil.copy2(best_weights, dest)
        print(f"Best weights saved to: {dest}")

    print(f"Training complete. Results: {results}")


if __name__ == "__main__":
    main()
