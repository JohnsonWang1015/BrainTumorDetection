from __future__ import annotations

import argparse
from pathlib import Path


def train_yolov11(
    task: str, data_yaml: Path, model: str, epochs: int, imgsz: int, project: Path
) -> None:
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(
        task=task, data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(project)
    )


def validate_yolov11(task: str, model_path: Path, data_yaml: Path) -> None:
    from ultralytics import YOLO

    yolo = YOLO(str(model_path))
    yolo.val(task=task, data=str(data_yaml))


def predict_yolov11(task: str, model_path: Path, source: Path, project: Path) -> None:
    from ultralytics import YOLO

    yolo = YOLO(str(model_path))
    yolo.predict(task=task, source=str(source), project=str(project), save=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv11 helper for segment/detect/classify"
    )
    parser.add_argument("--mode", choices=["train", "val", "predict"], required=True)
    parser.add_argument(
        "--task", choices=["segment", "detect", "classify"], default="segment"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("artifacts/yolo/dataset.yaml")
    )
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt")
    parser.add_argument(
        "--model-path", type=Path, default=Path("runs/segment/train/weights/best.pt")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/yolo/images/val")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("runs/yolo11"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_yolov11(
            args.task, args.data, args.model, args.epochs, args.imgsz, args.project
        )
    elif args.mode == "val":
        validate_yolov11(args.task, args.model_path, args.data)
    else:
        predict_yolov11(args.task, args.model_path, args.source, args.project)


if __name__ == "__main__":
    main()
