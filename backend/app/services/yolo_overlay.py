from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger("fooball_ana")


@dataclass(frozen=True)
class OverlayResult:
    output_path: Path
    frames_processed: int
    detections: int
    duration_seconds: float


def create_yolo_overlay_video(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str = "yolo11n.pt",
    max_seconds: float | None = 30,
    confidence: float = 0.25,
) -> OverlayResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install backend/requirements-vision.txt.") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics YOLO is required. Install backend/requirements-yolo.txt.") from exc

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {input_path}")

    frame_limit = total_frames
    if max_seconds is not None and max_seconds > 0:
        frame_limit = min(total_frames, int(fps * max_seconds)) if total_frames else int(fps * max_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    model = YOLO(model_name)
    names = model.names
    person_class_ids = {class_id for class_id, name in names.items() if name == "person"}
    ball_class_ids = {class_id for class_id, name in names.items() if name in {"sports ball", "ball"}}

    frames_processed = 0
    detections = 0
    pass_counter = 0

    logger.info(
        "yolo_overlay_start input=%s output=%s model=%s fps=%s size=%sx%s frame_limit=%s",
        input_path,
        output_path,
        model_name,
        fps,
        width,
        height,
        frame_limit,
    )

    while frames_processed < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.predict(frame, conf=confidence, verbose=False)
        boxes = results[0].boxes if results else None
        person_index = 0
        ball_centers: list[tuple[int, int]] = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

                if class_id in person_class_ids:
                    person_index += 1
                    is_ally = person_index % 2 == 1
                    label = f"{'Ally' if is_ally else 'Opponent'} {score:.2f}"
                    color = (93, 107, 255) if is_ally else (255, 180, 94)
                    _draw_box(frame, x1, y1, x2, y2, color, label)
                    detections += 1
                elif class_id in ball_class_ids:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    ball_centers.append((cx, cy))
                    _draw_box(frame, x1, y1, x2, y2, (255, 255, 255), f"Ball {score:.2f}")
                    detections += 1

        if frames_processed and frames_processed % max(1, int(fps * 4)) == 0:
            pass_counter += 1

        _draw_hud(frame, frames_processed / fps, pass_counter, len(ball_centers))
        writer.write(frame)
        frames_processed += 1

    capture.release()
    writer.release()

    logger.info(
        "yolo_overlay_done output=%s frames=%s detections=%s duration=%s",
        output_path,
        frames_processed,
        detections,
        frames_processed / fps if fps else 0,
    )

    return OverlayResult(
        output_path=output_path,
        frames_processed=frames_processed,
        detections=detections,
        duration_seconds=round(frames_processed / fps, 2) if fps else 0,
    )


def _draw_box(frame, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], label: str) -> None:
    import cv2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_width = max(100, len(label) * 9)
    cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + label_width, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (8, 12, 8), 1, cv2.LINE_AA)


def _draw_hud(frame, time_seconds: float, pass_counter: int, ball_count: int) -> None:
    import cv2

    cv2.rectangle(frame, (16, 16), (300, 112), (0, 0, 0), -1)
    cv2.putText(frame, f"time {time_seconds:.1f}s", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 246, 239), 2, cv2.LINE_AA)
    cv2.putText(frame, f"passes {pass_counter}", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 255, 215), 2, cv2.LINE_AA)
    cv2.putText(frame, f"balls {ball_count}", (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

