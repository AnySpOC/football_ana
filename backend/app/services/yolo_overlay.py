from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import logging
import math

logger = logging.getLogger("fooball_ana")


@dataclass(frozen=True)
class OverlayResult:
    output_path: Path
    timeline_path: Path
    frames_processed: int
    detections: int
    duration_seconds: float


@dataclass(frozen=True)
class TeamVisual:
    label: str
    box_color: tuple[int, int, int]
    center_bgr: tuple[float, float, float]


class TeamColorClassifier:
    def __init__(self, team_a: TeamVisual | None, team_b: TeamVisual | None, unknown: TeamVisual) -> None:
        self.team_a = team_a
        self.team_b = team_b
        self.unknown = unknown

    def classify(self, frame, x1: int, y1: int, x2: int, y2: int) -> TeamVisual:
        jersey_color = _extract_jersey_bgr(frame, x1, y1, x2, y2)
        if jersey_color is None or self.team_a is None or self.team_b is None:
            return self.unknown

        team_a_distance = _bgr_distance(jersey_color, self.team_a.center_bgr)
        team_b_distance = _bgr_distance(jersey_color, self.team_b.center_bgr)
        return self.team_a if team_a_distance <= team_b_distance else self.team_b


@dataclass
class TeamAssignment:
    team: TeamVisual
    disagreement_count: int = 0


class TeamAssignmentSmoother:
    def __init__(self, *, unknown: TeamVisual, switch_frames: int = 8) -> None:
        self.unknown = unknown
        self.switch_frames = switch_frames
        self.assignments: dict[int, TeamAssignment] = {}

    def resolve(self, track_id: int, observed_team: TeamVisual) -> TeamVisual:
        assignment = self.assignments.get(track_id)
        if assignment is None:
            assignment = TeamAssignment(team=observed_team)
            self.assignments[track_id] = assignment
            return assignment.team

        if observed_team.label == "unknown":
            return assignment.team
        if assignment.team.label == "unknown" or observed_team.label == assignment.team.label:
            assignment.team = observed_team
            assignment.disagreement_count = 0
            return assignment.team

        assignment.disagreement_count += 1
        if assignment.disagreement_count >= self.switch_frames:
            assignment.team = observed_team
            assignment.disagreement_count = 0
        return assignment.team


def create_yolo_overlay_video(
    input_path: Path,
    output_path: Path,
    *,
    timeline_path: Path | None = None,
    model_name: str = "yolo11n.pt",
    max_seconds: float | None = 30,
    confidence: float = 0.25,
    ball_confidence: float = 0.12,
    image_size: int = 1280,
    ally_color: str | None = None,
    swap_teams: bool = False,
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
    timeline_path = timeline_path or output_path.with_name(f"{output_path.stem}_timeline.json")
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
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
    team_classifier = _build_team_classifier(
        capture=capture,
        model=model,
        frame_limit=frame_limit,
        fps=fps,
        person_class_ids=person_class_ids,
        confidence=confidence,
        ally_color=ally_color,
        swap_teams=swap_teams,
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    team_smoother = TeamAssignmentSmoother(
        unknown=team_classifier.unknown,
        switch_frames=max(4, int(fps * 0.25)),
    )

    frames_processed = 0
    detections = 0
    timeline_frames: list[dict[str, object]] = []

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

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=min(confidence, ball_confidence),
            imgsz=image_size,
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        frame_objects: list[dict[str, object]] = []
        player_count = 0
        ball_count = 0

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                track_id = int(box.id[0]) if box.id is not None else None

                if class_id in person_class_ids and score >= confidence:
                    observed_team = team_classifier.classify(frame, x1, y1, x2, y2)
                    team = team_smoother.resolve(track_id, observed_team) if track_id is not None else observed_team
                    display_team = team.label.replace("_", " ").title()
                    display_id = f" #{track_id}" if track_id is not None else ""
                    _draw_box(frame, x1, y1, x2, y2, team.box_color, f"{display_team}{display_id} {score:.2f}")
                    frame_objects.append(
                        {
                            "kind": "player",
                            "track_id": track_id,
                            "team": team.label,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(score, 4),
                        }
                    )
                    player_count += 1
                    detections += 1
                elif class_id in ball_class_ids and score >= ball_confidence:
                    _draw_box(frame, x1, y1, x2, y2, (255, 255, 255), f"Ball {score:.2f}")
                    frame_objects.append(
                        {
                            "kind": "ball",
                            "track_id": track_id,
                            "team": None,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(score, 4),
                        }
                    )
                    ball_count += 1
                    detections += 1

        timeline_frames.append(
            {
                "frame_index": frames_processed,
                "timestamp_ms": round(frames_processed / fps * 1000) if fps else 0,
                "objects": frame_objects,
            }
        )
        _draw_hud(frame, frames_processed / fps, player_count, ball_count)
        writer.write(frame)
        frames_processed += 1

    capture.release()
    writer.release()

    timeline = {
        "schema_version": 1,
        "video": {
            "fps": round(fps, 4),
            "width": width,
            "height": height,
            "duration_seconds": round(frames_processed / fps, 3) if fps else 0,
        },
        "teams": [
            {"id": "team_a", "display_name": "Team A"},
            {"id": "team_b", "display_name": "Team B"},
            {"id": "unknown", "display_name": "Unknown"},
        ],
        "frames": timeline_frames,
    }
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    logger.info(
        "yolo_overlay_done output=%s timeline=%s frames=%s detections=%s duration=%s",
        output_path,
        timeline_path,
        frames_processed,
        detections,
        frames_processed / fps if fps else 0,
    )

    return OverlayResult(
        output_path=output_path,
        timeline_path=timeline_path,
        frames_processed=frames_processed,
        detections=detections,
        duration_seconds=round(frames_processed / fps, 2) if fps else 0,
    )


def _build_team_classifier(
    *,
    capture,
    model,
    frame_limit: int,
    fps: float,
    person_class_ids: set[int],
    confidence: float,
    ally_color: str | None,
    swap_teams: bool,
) -> TeamColorClassifier:
    import cv2
    import numpy as np

    max_scan_frames = min(frame_limit, max(1, int(fps * 8)))
    stride = max(1, int(fps // 2) or 1)
    samples: list[tuple[float, float, float]] = []

    for frame_index in range(0, max_scan_frames, stride):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        results = model.predict(frame, conf=confidence, imgsz=1280, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            if class_id not in person_class_ids:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            jersey_color = _extract_jersey_bgr(frame, x1, y1, x2, y2)
            if jersey_color is not None:
                samples.append(jersey_color)

    if len(samples) < 2:
        logger.warning("team_color_samples_insufficient count=%s fallback=unknown", len(samples))
        return TeamColorClassifier(
            team_a=None,
            team_b=None,
            unknown=TeamVisual("unknown", (160, 160, 160), (0.0, 0.0, 0.0)),
        )

    sample_array = np.float32(samples)
    compactness, labels, centers = cv2.kmeans(
        sample_array,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2),
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    centers_list = [tuple(float(value) for value in center) for center in centers]
    label_counts = [int((labels == index).sum()) for index in range(2)]

    if ally_color:
        target = _parse_color_to_bgr(ally_color)
        team_a_index = 0 if _bgr_distance(centers_list[0], target) <= _bgr_distance(centers_list[1], target) else 1
    else:
        team_a_index = 0 if label_counts[0] >= label_counts[1] else 1
    team_b_index = 1 - team_a_index
    if swap_teams:
        team_a_index, team_b_index = team_b_index, team_a_index

    logger.info(
        "team_color_clusters centers=%s counts=%s compactness=%s team_a_index=%s swap=%s",
        centers_list,
        label_counts,
        compactness,
        team_a_index,
        swap_teams,
    )

    return TeamColorClassifier(
        team_a=TeamVisual("team_a", (93, 107, 255), centers_list[team_a_index]),
        team_b=TeamVisual("team_b", (255, 180, 94), centers_list[team_b_index]),
        unknown=TeamVisual("unknown", (160, 160, 160), (0.0, 0.0, 0.0)),
    )


def _extract_jersey_bgr(frame, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float, float] | None:
    import cv2

    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 8 or box_height < 16:
        return None

    torso_x1 = x1 + int(box_width * 0.2)
    torso_x2 = x2 - int(box_width * 0.2)
    torso_y1 = y1 + int(box_height * 0.18)
    torso_y2 = y1 + int(box_height * 0.55)
    roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45)
    if int(mask.sum()) < 12:
        mask = hsv[:, :, 2] > 35
    if int(mask.sum()) < 12:
        return None

    pixels = roi[mask]
    mean = pixels.reshape(-1, 3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _bgr_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _parse_color_to_bgr(value: str) -> tuple[float, float, float]:
    named = {
        "red": (40.0, 40.0, 210.0),
        "blue": (210.0, 90.0, 40.0),
        "green": (70.0, 180.0, 70.0),
        "yellow": (40.0, 210.0, 210.0),
        "white": (230.0, 230.0, 230.0),
        "black": (25.0, 25.0, 25.0),
    }
    normalized = value.strip().lower()
    if normalized in named:
        return named[normalized]
    if normalized.startswith("#") and len(normalized) == 7:
        red = int(normalized[1:3], 16)
        green = int(normalized[3:5], 16)
        blue = int(normalized[5:7], 16)
        return (float(blue), float(green), float(red))
    raise ValueError(f"Unsupported ally color: {value}")


def _draw_box(frame, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], label: str) -> None:
    import cv2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_width = max(100, len(label) * 9)
    cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + label_width, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (8, 12, 8), 1, cv2.LINE_AA)


def _draw_hud(frame, time_seconds: float, player_count: int, ball_count: int) -> None:
    import cv2

    cv2.rectangle(frame, (16, 16), (300, 112), (0, 0, 0), -1)
    cv2.putText(frame, f"time {time_seconds:.1f}s", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 246, 239), 2, cv2.LINE_AA)
    cv2.putText(frame, f"players {player_count}", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 255, 215), 2, cv2.LINE_AA)
    cv2.putText(frame, f"ball detections {ball_count}", (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
