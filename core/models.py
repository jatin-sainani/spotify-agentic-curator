"""Pydantic models for spotify-agentic-curator."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Track(BaseModel):
    """Represents a Spotify track."""

    id: str = Field(..., description="Spotify track ID")
    name: str = Field(..., description="Track name")
    artist: str = Field(..., description="Artist name")
    album: str = Field(..., description="Album name")
    duration_ms: int = Field(..., description="Track duration in milliseconds")
    popularity: Optional[int] = Field(None, description="Track popularity score")
    external_url: Optional[str] = Field(None, description="Spotify external URL")
    preview_url: Optional[str] = Field(None, description="30-second preview URL")

    model_config = ConfigDict(populate_by_name=True)


class FeedbackType(str, Enum):
    """Types of feedback events."""

    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"
    REPEAT = "repeat"
    ADD_TO_PLAYLIST = "add_to_playlist"


class FeedbackEvent(BaseModel):
    """Represents user feedback on a track."""

    track_id: str = Field(..., description="ID of the track being rated")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    timestamp: datetime = Field(default_factory=datetime.now, description="When feedback occurred")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    model_config = ConfigDict(populate_by_name=True)


class IntentType(str, Enum):
    """Types of user intents for curation."""

    DISCOVER_NEW_MUSIC = "discover_new_music"
    CREATE_PLAYLIST = "create_playlist"
    IMPROVE_EXISTING_PLAYLIST = "improve_existing_playlist"
    EXPLORE_GENRE = "explore_genre"
    MOOD_BASED = "mood_based"


class Intent(BaseModel):
    """Represents user intent for music curation."""

    type: IntentType = Field(..., description="Type of intent")
    description: str = Field(..., description="Human-readable description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent-specific parameters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    model_config = ConfigDict(populate_by_name=True)


class PlanAction(str, Enum):
    """Types of actions in a curation plan."""

    SEARCH_TRACKS = "search_tracks"
    FILTER_BY_GENRE = "filter_by_genre"
    FILTER_BY_POPULARITY = "filter_by_popularity"
    SORT_BY_FEATURES = "sort_by_features"
    DIVERSIFY_ARTISTS = "diversify_artists"
    BALANCE_ENERGY = "balance_energy"


class PlanStep(BaseModel):
    """A single step in a curation plan."""

    action: PlanAction = Field(..., description="Type of action to perform")
    description: str = Field(..., description="Description of the step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    order: int = Field(..., description="Execution order")

    model_config = ConfigDict(populate_by_name=True)


class Draft(BaseModel):
    """Represents a playlist draft during curation."""

    id: str = Field(..., description="Unique draft ID")
    name: str = Field(..., description="Draft playlist name")
    description: Optional[str] = Field(None, description="Draft description")
    tracks: List[Track] = Field(default_factory=list, description="Tracks in the draft")
    intent: Intent = Field(..., description="Original user intent")
    plan: List[PlanStep] = Field(default_factory=list, description="Curation plan steps")
    feedback_events: List[FeedbackEvent] = Field(default_factory=list, description="User feedback")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    model_config = ConfigDict(populate_by_name=True)
