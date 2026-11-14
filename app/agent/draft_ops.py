"""Draft editing operations (add / remove / move / rerank / export).

Pure functions over Draft objects, so the CLI, the improve loop and (later) the
MCP server share one implementation. Every edit appends a feedback event to the
draft. These events are explicit user signals that evaluations can use.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.intent import IntentNormalized
from app.core.draft_models import Draft, TrackRef
from core.config import PROJECT_ROOT
from core.logging import logger

EXPORTS_DIR = PROJECT_ROOT / "exports"


class DraftError(ValueError):
    """User-facing error for an invalid draft operation."""


def _event(draft: Draft, kind: str, **data: Any) -> None:
    draft.feedback_events.append({"type": kind, "at": datetime.now().isoformat(timespec="seconds"), **data})


def _index_of(draft: Draft, track_id: str) -> int:
    for i, t in enumerate(draft.tracks):
        if t.id == track_id:
            return i
    raise DraftError(f"Track '{track_id}' is not in draft {draft.id}")


def _check_position(position: int, upper: int) -> int:
    """Validate a 1-based position against 1..upper and return the 0-based index."""
    if not 1 <= position <= upper:
        raise DraftError(f"Position must be between 1 and {upper}")
    return position - 1


def add_track(draft: Draft, track: TrackRef, position: Optional[int] = None) -> Draft:
    """Insert a track (1-based position; default: append)."""
    if any(t.id == track.id for t in draft.tracks):
        raise DraftError(f"Track '{track.id}' is already in draft {draft.id}")
    index = len(draft.tracks) if position is None else _check_position(position, len(draft.tracks) + 1)
    track.why = track.why or "Added by you"
    draft.tracks.insert(index, track)
    _event(draft, "add", track_id=track.id, position=index + 1)
    return draft


def remove_track(draft: Draft, track_id: str) -> TrackRef:
    """Remove a track and return it."""
    removed = draft.tracks.pop(_index_of(draft, track_id))
    _event(draft, "remove", track_id=track_id, name=removed.name)
    return removed


def move_track(draft: Draft, track_id: str, position: int) -> Draft:
    """Move a track to a new 1-based position."""
    src = _index_of(draft, track_id)
    dst = _check_position(position, len(draft.tracks))
    track = draft.tracks.pop(src)
    draft.tracks.insert(dst, track)
    _event(draft, "move", track_id=track_id, from_position=src + 1, to_position=dst + 1)
    return draft


def rerank(draft: Draft, criteria: str, use_llm: Optional[bool] = None) -> Draft:
    """Reorder a draft's tracks for the original request plus new criteria.

    Uses the LLM scorer when available; otherwise orders by audio-feature fit
    to the criteria parsed as an intent. Track membership never changes.
    """
    from app.agent.intent import parse_intent
    from app.agent.ranker import _llm_available, _stage1_pre_score, _stage2_llm_score

    if not draft.tracks:
        raise DraftError("Draft has no tracks to rerank")
    request = f"{draft.intent_text}. Re-order with this priority: {criteria}"
    intent = parse_intent(request, len(draft.tracks))
    if intent.mode != "curate":
        intent = IntentNormalized(mode="curate", description=request, constraints=intent.constraints,
                                  structure=intent.structure)

    stage1 = _stage1_pre_score(list(draft.tracks), intent, set())
    if use_llm is None:
        use_llm = _llm_available()
    if use_llm:
        scored = [(t, s) for t, s, _ in _stage2_llm_score(stage1, intent, request_text=request)]
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        scored = stage1  # already sorted
    before = [t.id for t in draft.tracks]
    draft.tracks = [t for t, _ in scored]
    _event(draft, "rerank", criteria=criteria, llm=bool(use_llm),
           moved=sum(1 for a, b in zip(before, (t.id for t in draft.tracks)) if a != b))
    return draft


def export_draft(draft: Draft, fmt: str = "uris", out_dir: Path = EXPORTS_DIR) -> Path:
    """Write a draft to a file.

    fmt:
      uris: one Spotify URI per line. Paste into a playlist in the Spotify
            desktop app to add every track at once.
      json: tracks with names, artists and explanations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "uris":
        path = out_dir / f"{draft.id}.txt"
        path.write_text("\n".join(t.uri for t in draft.tracks) + "\n", encoding="utf-8")
    elif fmt == "json":
        path = out_dir / f"{draft.id}.json"
        payload = {
            "id": draft.id,
            "request": draft.intent_text,
            "tracks": [{"uri": t.uri, "name": t.name, "artists": t.artists, "why": t.why} for t in draft.tracks],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        raise DraftError(f"Unknown export format '{fmt}' (expected 'uris' or 'json')")
    logger.info(f"Exported draft {draft.id} to {path}")
    return path


def confirm_to_spotify(draft: Draft, client, name: str, description: str = "", public: bool = False) -> Dict[str, Any]:
    """Create a Spotify playlist from a draft and mark the draft confirmed."""
    if not draft.tracks:
        raise DraftError("Draft has no tracks")
    playlist = client.create_playlist(name=name, description=description, public=public)
    client.add_items_to_playlist(playlist["id"], [t.uri for t in draft.tracks])
    draft.status = "confirmed"
    draft.external_url = playlist.get("external_urls", {}).get("spotify")
    _event(draft, "confirm", target="spotify", playlist_id=playlist["id"])
    return playlist


def resolve_tracks(catalog, track_ids: List[str]) -> Dict[str, TrackRef]:
    """Look up tracks by ID in a catalog, keyed by ID."""
    return {t.id: t for t in catalog.get_tracks(track_ids)}
