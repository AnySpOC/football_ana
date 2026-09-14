from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


DATASET_CLASS_NAMES = {
    "person": 0,
    "sports ball": 1,
    "ball": 1,
}


def main() -> None:
    parser = ArgumentParser(description="Create YOLO-format draft labels for extracted frames.")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/futsal"), help="Dataset root")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model or local .pt path")
    parser.add_argument("--person-conf", type=float, default=0.25, help="Person confidence threshold")
    parser.add_argument("--ball-conf", type=float, default=0.08, help="Ball confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install backend/requirements-vision.txt.") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics YOLO is required. Install backend/requirements-yolo.txt.") from exc

    model = YOLO(args.model)
    model_names = model.names
    source_class_ids = {
        class_id: DATASET_CLASS_NAMES[name]
        for class_id, name in model_names.items()
        if name in DATASET_CLASS_NAMES
    }

    total_images = 0
    total_labels = 0
    for split in ("train", "val"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(image_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            rows: list[str] = []
            results = model.predict(image, conf=min(args.person_conf, args.ball_conf), imgsz=args.imgsz, verbose=False)
            boxes = results[0].boxes if results else None
            if boxes is not None:
                for box in boxes:
                    source_class_id = int(box.cls[0])
                    if source_class_id not in source_class_ids:
                        continue
                    score = float(box.conf[0])
                    dataset_class_id = source_class_ids[source_class_id]
                    if dataset_class_id == 0 and score < args.person_conf:
                        continue
                    if dataset_class_id == 1 and score < args.ball_conf:
                        continue

                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    x1 = max(0.0, min(width, x1))
                    x2 = max(0.0, min(width, x2))
                    y1 = max(0.0, min(height, y1))
                    y2 = max(0.0, min(height, y2))
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width <= 1 or box_height <= 1:
                        continue

                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    norm_width = box_width / width
                    norm_height = box_height / height
                    rows.append(
                        f"{dataset_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )

            label_path = label_dir / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            total_images += 1
            total_labels += len(rows)

    print(f"dataset={args.dataset}")
    print(f"images={total_images}")
    print(f"labels={total_labels}")


if __name__ == "__main__":
    main()
