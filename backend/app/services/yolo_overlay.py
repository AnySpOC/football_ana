from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import math

logger = logging.getLogger("fooball_ana")


@dataclass(frozen=True)
class OverlayResult:
    output_path: Path
    frames_processed: int
    detections: int
    duration_seconds: float


@dataclass(frozen=True)
class TeamVisual:
    label: str
    box_color: tuple[int, int, int]
    center_bgr: tuple[float, float, float]


class TeamColorClassifier:
    def __init__(self, ally: TeamVisual, opponent: TeamVisual) -> None:
        self.ally = ally
        self.opponent = opponent

    def classify(self, frame, x1: int, y1: int, x2: int, y2: int) -> TeamVisual:
        jersey_color = _extract_jersey_bgr(frame, x1, y1, x2, y2)
        if jersey_color is None:
            return self.ally

        ally_distance = _bgr_distance(jersey_color, self.ally.center_bgr)
        opponent_distance = _bgr_distance(jersey_color, self.opponent.center_bgr)
        return self.ally if ally_distance <= opponent_distance else self.opponent


@dataclass
class TeamTrack:
    track_id: int
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    team: TeamVisual
    disagreement_count: int = 0
    missed_frames: int = 0


class TeamStabilizer:
    def __init__(self, *, max_distance: float, max_missed_frames: int, switch_frames: int = 8) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.switch_frames = switch_frames
        self.next_track_id = 1
        self.tracks: list[TeamTrack] = []

    def update(
        self,
        detections: list[tuple[int, int, int, int, float, TeamVisual]],
    ) -> list[tuple[int, int, int, int, float, TeamVisual, int]]:
        for track in self.tracks:
            track.missed_frames += 1

        assignments: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        for det_index, detection in enumerate(detections):
            center = _bbox_center(detection[:4])
            best_track_index = None
            best_distance = self.max_distance
            for track_index, track in enumerate(self.tracks):
                if track_index in used_tracks:
                    continue
                distance = _point_distance(center, track.center)
                if distance < best_distance:
                    best_distance = distance
                    best_track_index = track_index
            if best_track_index is not None:
                assignments.append((det_index, best_track_index))
                used_tracks.add(best_track_index)

        assigned_detections = {det_index for det_index, _ in assignments}
        output: list[tuple[int, int, int, int, float, TeamVisual, int]] = []

        for det_index, track_index in assignments:
            x1, y1, x2, y2, score, observed_team = detections[det_index]
            track = self.tracks[track_index]
            track.center = _bbox_center((x1, y1, x2, y2))
            track.bbox = (x1, y1, x2, y2)
            track.missed_frames = 0

            if observed_team.label == track.team.label:
                track.disagreement_count = 0
            else:
                track.disagreement_count += 1
                if track.disagreement_count >= self.switch_frames:
                    track.team = observed_team
                    track.disagreement_count = 0

            output.append((x1, y1, x2, y2, score, track.team, track.track_id))

        for det_index, detection in enumerate(detections):
            if det_index in assigned_detections:
                continue
            x1, y1, x2, y2, score, observed_team = detection
            track = TeamTrack(
                track_id=self.next_track_id,
                center=_bbox_center((x1, y1, x2, y2)),
                bbox=(x1, y1, x2, y2),
                team=observed_team,
            )
            self.next_track_id += 1
            self.tracks.append(track)
            output.append((x1, y1, x2, y2, score, track.team, track.track_id))

        self.tracks = [track for track in self.tracks if track.missed_frames <= self.max_missed_frames]
        return output


def create_yolo_overlay_video(
    input_path: Path,
    output_path: Path,
    *,
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
    team_stabilizer = TeamStabilizer(
        max_distance=max(48.0, width * 0.055),
        max_missed_frames=max(4, int(fps * 0.7)),
        switch_frames=max(4, int(fps * 0.25)),
    )

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

        results = model.predict(frame, conf=min(confidence, ball_confidence), imgsz=image_size, verbose=False)
        boxes = results[0].boxes if results else None
        ball_centers: list[tuple[int, int]] = []

        person_detections: list[tuple[int, int, int, int, float, TeamVisual]] = []
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

                if class_id in person_class_ids and score >= confidence:
                    observed_team = team_classifier.classify(frame, x1, y1, x2, y2)
                    person_detections.append((x1, y1, x2, y2, score, observed_team))
                elif class_id in ball_class_ids and score >= ball_confidence:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    ball_centers.append((cx, cy))
                    _draw_box(frame, x1, y1, x2, y2, (255, 255, 255), f"Ball {score:.2f}")
                    detections += 1

        for x1, y1, x2, y2, score, team, track_id in team_stabilizer.update(person_detections):
            label = f"{team.label} #{track_id} {score:.2f}"
            _draw_box(frame, x1, y1, x2, y2, team.box_color, label)
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



def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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
        logger.warning("team_color_samples_insufficient count=%s fallback=red_blue", len(samples))
        return TeamColorClassifier(
            ally=TeamVisual("Ally", (93, 107, 255), (70.0, 70.0, 210.0)),
            opponent=TeamVisual("Opponent", (255, 180, 94), (210.0, 120.0, 60.0)),
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
        ally_index = 0 if _bgr_distance(centers_list[0], target) <= _bgr_distance(centers_list[1], target) else 1
    else:
        ally_index = 0 if label_counts[0] >= label_counts[1] else 1
    opponent_index = 1 - ally_index
    if swap_teams:
        ally_index, opponent_index = opponent_index, ally_index

    logger.info(
        "team_color_clusters centers=%s counts=%s compactness=%s ally_index=%s swap=%s",
        centers_list,
        label_counts,
        compactness,
        ally_index,
        swap_teams,
    )

    return TeamColorClassifier(
        ally=TeamVisual("Ally", (93, 107, 255), centers_list[ally_index]),
        opponent=TeamVisual("Opponent", (255, 180, 94), centers_list[opponent_index]),
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


def _draw_hud(frame, time_seconds: float, pass_counter: int, ball_count: int) -> None:
    import cv2

    cv2.rectangle(frame, (16, 16), (300, 112), (0, 0, 0), -1)
    cv2.putText(frame, f"time {time_seconds:.1f}s", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 246, 239), 2, cv2.LINE_AA)
    cv2.putText(frame, f"passes {pass_counter}", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 255, 215), 2, cv2.LINE_AA)
    cv2.putText(frame, f"balls {ball_count}", (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
