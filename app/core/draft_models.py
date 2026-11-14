"""Draft-specific Pydantic models for playlist curation."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrackRef(BaseModel):
    """Reference to a Spotify track in a draft."""

    uri: str = Field(..., description="Spotify track URI")
    name: str = Field(..., description="Track name")
    artists: List[str] = Field(..., description="List of artist names")
    id: str = Field(..., description="Spotify track ID")
    audio_features: Optional[Dict[str, Any]] = Field(None, description="Audio features dictionary")
    genres: List[str] = Field(default_factory=list, description="Genre labels (broad first, then niche)")
    year: Optional[int] = Field(None, description="Release year")
    popularity: Optional[int] = Field(None, description="Popularity 0-100")
    why: Optional[str] = Field(None, description="Reason for inclusion in draft")

    model_config = ConfigDict(populate_by_name=True)


class DraftStatus(str, Enum):
    """Draft status enumeration."""

    CREATED = "created"
    CURATING = "curating"
    COMPLETED = "completed"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class Draft(BaseModel):
    """Represents a playlist draft during curation."""

    id: str = Field(..., description="Unique draft ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    intent_text: str = Field(..., description="Original user intent/prompt text")
    mode: Literal["curate", "top_tracks", "library"] = Field(..., description="Curation mode")
    length: int = Field(..., description="Target playlist length")
    tracks: List[TrackRef] = Field(default_factory=list, description="Tracks in the draft")
    feedback_events: List[Dict[str, Any]] = Field(default_factory=list, description="User feedback events")
    status: DraftStatus = Field(default=DraftStatus.CREATED, description="Current draft status")
    external_url: Optional[str] = Field(None, description="Spotify playlist URL once confirmed")

    model_config = ConfigDict(populate_by_name=True)
