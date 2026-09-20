from pathlib import Path
import logging

from ..schemas import AnalysisSummary, TeamSummary, VideoMetadata

logger = logging.getLogger("fooball_ana")


def analyze_video(job_id: str, video_path: Path, result_dir: Path) -> AnalysisSummary:
    metadata = _read_video_metadata(video_path)
    notes = [
        "動画メタ情報を取得しました。人物・ボール検出は次の解析ステップで実行します。",
        "ポゼッション、パス、シュート、距離は未計測のためnullです。",
    ]

    if metadata.frame_count == 0:
        notes.insert(
            0,
            "動画メタ情報を取得できませんでした。OpenCVが未導入、または動画コーデックを読めない可能性があります。",
        )

    summary = AnalysisSummary(
        job_id=job_id,
        metadata=metadata,
        teams=[
            TeamSummary(
                name="Team A",
                color_hex="#e53935",
                possession_rate=None,
                estimated_passes=None,
                estimated_shots=None,
            ),
            TeamSummary(
                name="Team B",
                color_hex="#1e88e5",
                possession_rate=None,
                estimated_passes=None,
                estimated_shots=None,
            ),
        ],
        players=[],
        ball_touch_candidates=None,
        notes=notes,
    )

    result_path = result_dir / f"{job_id}.json"
    result_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def _read_video_metadata(video_path: Path) -> VideoMetadata:
    try:
        import cv2
    except ImportError as exc:
        logger.exception("opencv_import_failed path=%s error=%s", video_path, exc)
        return VideoMetadata(
            duration_seconds=0,
            fps=0,
            width=0,
            height=0,
            frame_count=0,
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.warning(
            "video_open_failed path=%s exists=%s size_bytes=%s",
            video_path,
            video_path.exists(),
            video_path.stat().st_size if video_path.exists() else 0,
        )
        return VideoMetadata(
            duration_seconds=0,
            fps=0,
            width=0,
            height=0,
            frame_count=0,
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_seconds = frame_count / fps if fps else 0
    logger.info(
        "video_metadata path=%s duration=%s fps=%s width=%s height=%s frames=%s",
        video_path,
        duration_seconds,
        fps,
        width,
        height,
        frame_count,
    )
    return VideoMetadata(
        duration_seconds=round(duration_seconds, 2),
        fps=round(fps, 2),
        width=width,
        height=height,
        frame_count=frame_count,
    )
