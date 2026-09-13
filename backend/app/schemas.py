from pydantic import BaseModel, Field


class TeamSummary(BaseModel):
    name: str
    color_hex: str
    possession_rate: float = Field(ge=0, le=1)
    estimated_passes: int = Field(ge=0)
    estimated_shots: int = Field(ge=0)


class PlayerSummary(BaseModel):
    track_id: str
    team: str
    distance_meters: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class VideoMetadata(BaseModel):
    duration_seconds: float = Field(ge=0)
    fps: float = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    frame_count: int = Field(ge=0)


class AnalysisSummary(BaseModel):
    job_id: str
    metadata: VideoMetadata
    teams: list[TeamSummary]
    players: list[PlayerSummary]
    ball_touch_candidates: int = Field(ge=0)
    notes: list[str]


class AnalysisJob(BaseModel):
    id: str
    filename: str
    status: str
    summary: AnalysisSummary


class YoloOverlayRequest(BaseModel):
    seconds: float = Field(default=30, ge=0, description="Max seconds to process. Use 0 for the full video.")
    confidence: float = Field(default=0.25, ge=0, le=1, description="YOLO detection confidence threshold.")
    ally_color: str | None = Field(default=None, description="Optional ally uniform color name or #RRGGBB.")


class YoloOverlayResponse(BaseModel):
    job_id: str
    overlay_url: str
    frames_processed: int = Field(ge=0)
    detections: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
