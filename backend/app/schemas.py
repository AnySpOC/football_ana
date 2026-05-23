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

