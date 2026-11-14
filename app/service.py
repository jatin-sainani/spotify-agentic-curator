"""Curator operations for agent clients (the MCP server is a thin adapter over this).

Every method returns JSON-serializable dicts and raises ValueError with a
message meant for the calling agent. Refinement is split across two calls,
propose_changes then apply_changes, so the calling agent can show the swaps to
its user and only apply what they approve: human-in-the-loop over MCP.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from app.agent.draft_ops import DraftError, add_track, export_draft, move_track, remove_track
from app.agent.draft_store import DRAFTS_DIR, list_drafts, load_draft, new_draft_id, save_draft
from app.core.draft_models import Draft, TrackRef
from core.logging import logger

MAX_LENGTH = 50


def track_summary(track: TrackRef, position: Optional[int] = None) -> Dict[str, Any]:
    """Compact, client-facing view of a track (metadata is sanitized for display)."""
    from app.agent.safety import sanitize_track

    clean, _ = sanitize_track(track)
    out = {"id": track.id, "uri": track.uri, "name": clean.name, "artists": clean.artists,
           "genres": clean.genres[:3], "year": track.year}
    if position is not None:
        out["position"] = position
    if track.why:
        out["why"] = track.why
    return out


class CuratorService:
    def __init__(self, catalog_factory: Optional[Callable[[], Any]] = None):
        """
        Args:
            catalog_factory: zero-arg callable returning a CatalogProvider
                (default: the configured catalog). Created lazily, once.
        """
        self._catalog_factory = catalog_factory
        self._catalog = None

    @property
    def catalog(self):
        if self._catalog is None:
            if self._catalog_factory is None:
                from app.catalog import get_catalog

                self._catalog_factory = get_catalog
            self._catalog = self._catalog_factory()
        return self._catalog

    @staticmethod
    def _load(draft_id: str) -> Draft:
        if not str(draft_id).isalnum():  # never let an agent-supplied id become a path
            raise ValueError(f"Invalid draft_id '{draft_id}'")
        try:
            return load_draft(draft_id)
        except FileNotFoundError:
            raise ValueError(f"Unknown draft_id '{draft_id}'. Use list_drafts to see existing drafts.")

    @staticmethod
    def _view(draft: Draft) -> Dict[str, Any]:
        return {
            "draft_id": draft.id,
            "request": draft.intent_text,
            "status": getattr(draft.status, "value", draft.status),
            "spotify_url": draft.external_url,
            "tracks": [track_summary(t, i) for i, t in enumerate(draft.tracks, 1)],
        }

    # -- curation -----------------------------------------------------------

    def curate_playlist(self, request: str, length: int = 15, retrieval: str = "auto",
                        sequence: bool = False) -> Dict[str, Any]:
        from app.agent.graph import run_curation

        request = (request or "").strip()
        if not request:
            raise ValueError("request must describe the playlist, e.g. 'mellow 70s soul for a rainy morning'")
        if not 5 <= length <= MAX_LENGTH:
            raise ValueError(f"length must be between 5 and {MAX_LENGTH}")
        if retrieval not in ("auto", "agent", "heuristic"):
            raise ValueError("retrieval must be 'auto', 'agent' or 'heuristic'")

        state = run_curation(request, length, self.catalog, retrieval=retrieval, sequence=sequence)
        intent = state.get("intent")
        draft = Draft(id=new_draft_id(), intent_text=request, mode=intent.mode if intent else "curate",
                      length=length, tracks=state.get("selected_tracks", []), status="created")
        save_draft(draft)
        view = self._view(draft)
        view.update({
            "retrieval_used": state.get("retrieval_used"),
            "catalog_searches": [c.get("query") for c in state.get("retrieval_log") or [] if c.get("query")],
            "interpreted_as": intent.description if intent else None,
        })
        return view

    def get_draft(self, draft_id: str) -> Dict[str, Any]:
        return self._view(self._load(draft_id))

    def list_drafts(self, limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for draft_id in list_drafts()[-limit:]:
            try:
                d = load_draft(draft_id)
                out.append({"draft_id": d.id, "request": d.intent_text, "tracks": len(d.tracks),
                            "created": d.created_at.isoformat(timespec="seconds")})
            except Exception as e:  # skip unreadable files, don't fail the listing
                logger.warning(f"Skipping draft {draft_id}: {e}")
        return out

    def explain_track(self, draft_id: str, track_id: str) -> Dict[str, Any]:
        draft = self._load(draft_id)
        for i, t in enumerate(draft.tracks, 1):
            if t.id == track_id:
                f = t.audio_features or {}
                return {**track_summary(t, i), "audio_features": {
                    k: round(f[k], 3) for k in ("energy", "valence", "danceability", "acousticness", "tempo") if k in f
                }, "popularity": t.popularity}
        raise ValueError(f"Track '{track_id}' is not in draft {draft_id}")

    def search_catalog(self, query: str, limit: int = 10, energy_max: Optional[float] = None,
                       energy_min: Optional[float] = None, year_min: Optional[int] = None,
                       year_max: Optional[int] = None) -> List[Dict[str, Any]]:
        from app.agent.tools import _filters

        filters = _filters(energy_min, energy_max, None, None, None, None, year_min, year_max)
        refs = self.catalog.search(query, limit=max(1, min(limit, 50)), filters=filters)
        return [track_summary(r) for r in refs]

    # -- editing ------------------------------------------------------------------

    def edit_draft(self, draft_id: str, action: str, track_id: str, position: Optional[int] = None) -> Dict[str, Any]:
        draft = self._load(draft_id)
        try:
            if action == "remove":
                remove_track(draft, track_id)
            elif action == "move":
                if position is None:
                    raise ValueError("position is required for action 'move'")
                move_track(draft, track_id, position)
            elif action == "add":
                found = self.catalog.get_tracks([track_id])
                if not found:
                    raise ValueError(f"Track '{track_id}' not found in the catalog")
                add_track(draft, found[0], position)
            else:
                raise ValueError("action must be 'add', 'remove' or 'move'")
        except DraftError as e:
            raise ValueError(str(e))
        save_draft(draft)
        return self._view(draft)

    def propose_changes(self, draft_id: str, feedback: str, max_swaps: int = 3) -> Dict[str, Any]:
        """Step 1 of refinement: propose swaps (nothing changes until apply_changes)."""
        from app.agent.improve import propose_swaps

        draft = self._load(draft_id)
        if not feedback.strip():
            raise ValueError("feedback must say what to change, e.g. 'fewer covers, more 80s'")
        proposals = propose_swaps(draft, feedback, self.catalog, max_swaps=max(1, min(max_swaps, 10)))
        _pending_path(draft_id).write_text(json.dumps({"feedback": feedback, "proposals": proposals}),
                                           encoding="utf-8")
        return {
            "draft_id": draft_id,
            "feedback": feedback,
            "proposals": [{
                "proposal_id": i,
                "remove": track_summary(TrackRef(**p["remove"])),
                "add": track_summary(TrackRef(**p["add"])),
                "score_gain": round(p["new_score"] - p["old_score"], 3),
            } for i, p in enumerate(proposals)],
            "next_step": "Show these to the user, then call apply_changes with the accepted proposal_ids."
                         if proposals else "No better-fitting replacements found; the draft is unchanged.",
        }

    def apply_changes(self, draft_id: str, accepted_proposal_ids: List[int]) -> Dict[str, Any]:
        """Step 2 of refinement: apply the accepted proposals, reject the rest."""
        from app.agent.improve import apply_swaps

        path = _pending_path(draft_id)
        if not path.exists():
            raise ValueError(f"No pending proposals for draft {draft_id}; call propose_changes first")
        pending = json.loads(path.read_text(encoding="utf-8"))
        proposals = pending["proposals"]
        bad = [i for i in accepted_proposal_ids if not 0 <= i < len(proposals)]
        if bad:
            raise ValueError(f"Unknown proposal_ids {bad}; valid ids are 0..{len(proposals) - 1}")
        decisions = [i in set(accepted_proposal_ids) for i in range(len(proposals))]
        draft, applied = apply_swaps(self._load(draft_id), proposals, decisions, pending["feedback"])
        save_draft(draft)
        path.unlink()
        return {**self._view(draft), "applied": applied, "rejected": len(proposals) - applied}

    def export(self, draft_id: str, fmt: str = "uris") -> Dict[str, Any]:
        try:
            path = export_draft(self._load(draft_id), fmt=fmt)
        except DraftError as e:
            raise ValueError(str(e))
        return {"draft_id": draft_id, "format": fmt, "path": str(path),
                "content": path.read_text(encoding="utf-8")}


def _pending_path(draft_id: str):
    if not draft_id.isalnum():  # draft ids are short hex strings; never build paths from anything else
        raise ValueError(f"Invalid draft_id '{draft_id}'")
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    return DRAFTS_DIR / f"{draft_id}.proposals.json"
