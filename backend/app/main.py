from pathlib import Path
import logging
import sys
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .schemas import AnalysisJob, AnalysisSummary, YoloOverlayRequest, YoloOverlayResponse
from .services.analyzer import analyze_video
from .services.yolo_overlay import create_yolo_overlay_video

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
LOG_DIR = DATA_DIR / "logs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("fooball_ana")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / "backend.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

app = FastAPI(title="Fooball Ana API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/api/videos", response_model=AnalysisJob)
async def upload_video(file: UploadFile = File(...)) -> AnalysisJob:
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイル名がありません")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="動画ファイルをアップロードしてください")

    job_id = uuid4().hex
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    logger.info("upload_start job_id=%s filename=%s content_type=%s", job_id, file.filename, file.content_type)
    with video_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    logger.info("upload_saved job_id=%s path=%s size_bytes=%s", job_id, video_path, video_path.stat().st_size)
    summary = analyze_video(job_id=job_id, video_path=video_path, result_dir=RESULT_DIR)
    logger.info(
        "analysis_completed job_id=%s duration=%s fps=%s width=%s height=%s frames=%s",
        job_id,
        summary.metadata.duration_seconds,
        summary.metadata.fps,
        summary.metadata.width,
        summary.metadata.height,
        summary.metadata.frame_count,
    )

    return AnalysisJob(
        id=job_id,
        filename=file.filename,
        status="completed",
        summary=summary,
    )


@app.get("/api/results/{job_id}", response_model=AnalysisSummary)
def get_result(job_id: str) -> AnalysisSummary:
    result_path = RESULT_DIR / f"{job_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="解析結果が見つかりません")

    return AnalysisSummary.model_validate_json(result_path.read_text(encoding="utf-8"))


@app.get("/api/results/{job_id}/overlay")
def get_overlay(job_id: str) -> FileResponse:
    overlay_path = RESULT_DIR / f"{job_id}_overlay.mp4"
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="オーバーレイ動画はまだ作成されていません")

    return FileResponse(overlay_path, media_type="video/mp4")


@app.get("/api/results/{job_id}/timeline")
def get_timeline(job_id: str) -> FileResponse:
    timeline_path = RESULT_DIR / f"{job_id}_timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="解析タイムラインはまだ作成されていません")

    return FileResponse(timeline_path, media_type="application/json")


@app.post("/api/results/{job_id}/overlay/yolo", response_model=YoloOverlayResponse)
def create_yolo_overlay(job_id: str, request: YoloOverlayRequest = YoloOverlayRequest()) -> YoloOverlayResponse:
    matches = sorted(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="元動画が見つかりません")

    output_path = RESULT_DIR / f"{job_id}_overlay.mp4"
    timeline_path = RESULT_DIR / f"{job_id}_timeline.json"
    try:
        result = create_yolo_overlay_video(
            input_path=matches[0],
            output_path=output_path,
            timeline_path=timeline_path,
            max_seconds=None if request.seconds <= 0 else request.seconds,
            confidence=request.confidence,
            ball_confidence=request.ball_confidence,
            image_size=request.image_size,
            ally_color=request.ally_color,
            swap_teams=request.swap_teams,
        )
    except RuntimeError as exc:
        logger.exception("yolo_overlay_failed job_id=%s error=%s", job_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return YoloOverlayResponse(
        job_id=job_id,
        overlay_url=f"/api/results/{job_id}/overlay",
        timeline_url=f"/api/results/{job_id}/timeline",
        frames_processed=result.frames_processed,
        detections=result.detections,
        duration_seconds=result.duration_seconds,
    )

@app.get("/api/logs")
def get_logs(lines: int = 200) -> dict[str, list[str]]:
    log_path = LOG_DIR / "backend.log"
    if not log_path.exists():
        return {"lines": []}

    safe_lines = max(1, min(lines, 1000))
    return {"lines": log_path.read_text(encoding="utf-8").splitlines()[-safe_lines:]}


