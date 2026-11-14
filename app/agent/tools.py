"""LangChain tools the retrieval agent calls, over any CatalogProvider.

Tools are created per run around a RetrievalSession. Every track a tool
returns is recorded in the session pool, so the ranker only ever sees tracks
that actually came from the catalog: the LLM steers *what to look for*, it
cannot inject track IDs of its own.
"""

import json
import threading
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from app.agent.safety import wrap_catalog_data
from app.catalog.base import CatalogProvider, FeatureFilters
from app.core.draft_models import TrackRef
from core.logging import logger

MAX_RESULTS_PER_CALL = 25


@dataclass
class RetrievalSession:
    """Mutable per-run state shared by the tools.

    LangGraph's ToolNode runs the tool calls of one model message in parallel
    threads, so results arrive in completion order. Each call is buffered under
    its tool_call_id, and finalize() rebuilds pool/calls in the order the model
    issued the calls, which keeps runs reproducible (ranking breaks score ties by
    pool order).
    """

    catalog: CatalogProvider
    pool: Dict[str, TrackRef] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)  # track id -> query that found it
    calls: List[Dict] = field(default_factory=list)       # tool-call log for observability/evals
    _buffer: Dict[str, Tuple[Dict, List[TrackRef], str]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log_call(self, call_id: str, entry: Dict, refs: List[TrackRef] = (), source: str = "") -> None:
        """Record one tool call (thread-safe). Live views stay usable mid-run."""
        entry = {**entry, "call_id": call_id}
        with self._lock:
            self._buffer[call_id] = (entry, list(refs), source)
            self.calls.append(entry)
            self._add(refs, source)

    def _add(self, refs, source: str) -> None:
        for ref in refs:
            if ref.id not in self.pool:
                self.pool[ref.id] = ref
                self.sources[ref.id] = source

    def finalize(self, call_order: List[str]) -> None:
        """Rebuild pool and call log in the model's call order."""
        with self._lock:
            ordered = [cid for cid in call_order if cid in self._buffer]
            ordered += [cid for cid in self._buffer if cid not in set(ordered)]
            self.pool, self.sources, self.calls = {}, {}, []
            for cid in ordered:
                entry, refs, source = self._buffer[cid]
                self.calls.append(entry)
                self._add(refs, source)


def describe_track(ref: TrackRef) -> str:
    """Compact one-line description of a track for tool output."""
    from app.agent.safety import for_prompt

    ref = for_prompt(ref)
    f = ref.audio_features or {}
    feats = " ".join(f"{k}={f[k]:.2f}" for k in ("energy", "valence", "acousticness") if k in f)
    tempo = f" tempo={f['tempo']:.0f}" if "tempo" in f else ""
    genres = ", ".join(ref.genres[:3])
    year = f" {ref.year}" if ref.year else ""
    return f"[{ref.id}] {ref.name} - {', '.join(ref.artists)} ({genres}){year} {feats}{tempo}".strip()


def _filters(
    energy_min: Optional[float], energy_max: Optional[float],
    valence_min: Optional[float], valence_max: Optional[float],
    tempo_min: Optional[float], tempo_max: Optional[float],
    year_min: Optional[int] = None, year_max: Optional[int] = None,
) -> Optional[FeatureFilters]:
    f = FeatureFilters()
    if year_min is not None or year_max is not None:
        f["year_range"] = [int(year_min or 1900), int(year_max or 2100)]
    if energy_min is not None or energy_max is not None:
        f["energy_range"] = [energy_min or 0.0, 1.0 if energy_max is None else energy_max]
    if valence_min is not None or valence_max is not None:
        f["valence_range"] = [valence_min or 0.0, 1.0 if valence_max is None else valence_max]
    if tempo_min is not None or tempo_max is not None:
        f["bpm_range"] = [tempo_min or 0.0, 300.0 if tempo_max is None else tempo_max]
    return f or None


def create_retrieval_tools(session: RetrievalSession) -> List[BaseTool]:
    """Build the tools bound to one retrieval session."""
    catalog = session.catalog

    @tool
    def search_catalog(
        query: str,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        valence_min: Optional[float] = None,
        valence_max: Optional[float] = None,
        tempo_min: Optional[float] = None,
        tempo_max: Optional[float] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 15,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Search the music catalog by genre, artist, title or mood words.

        Use short queries (1-3 words) like "indie folk", "bossa nova", "Radiohead",
        "rain". Optionally restrict audio features (0-1 scales; tempo in BPM):
        energy (intensity), valence (musical positivity: low = sad, high = happy).
        For decades use release years, e.g. "80s" -> year_min=1980, year_max=1989.
        Returns one line per track: [track_id] title - artists (genres) year features.
        """
        limit = max(1, min(int(limit), MAX_RESULTS_PER_CALL))
        filters = _filters(energy_min, energy_max, valence_min, valence_max, tempo_min, tempo_max,
                           year_min, year_max)
        refs = catalog.search(query, limit=limit, filters=filters)
        session.log_call(tool_call_id, {"tool": "search_catalog", "query": query, "filters": dict(filters or {}),
                                        "results": len(refs), "ids": [r.id for r in refs]}, refs, f"search:{query}")
        logger.info(f"[agent] search_catalog({query!r}, filters={dict(filters or {})}) -> {len(refs)} tracks")
        if not refs:
            return "No tracks found. Try a broader or different query, or loosen the feature filters."
        return wrap_catalog_data("\n".join(describe_track(r) for r in refs))

    @tool
    def get_user_taste(limit: int = 15, tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Get tracks the listener already likes (their top tracks / favorite artists).

        Use this to understand their taste and to personalize the playlist.
        """
        limit = max(1, min(int(limit), MAX_RESULTS_PER_CALL))
        refs = catalog.get_user_taste(limit=limit)
        session.log_call(tool_call_id, {"tool": "get_user_taste", "results": len(refs),
                                        "ids": [r.id for r in refs]}, refs, "taste")
        logger.info(f"[agent] get_user_taste() -> {len(refs)} tracks")
        if not refs:
            return "No listening history available for this user."
        return wrap_catalog_data("\n".join(describe_track(r) for r in refs))

    @tool
    def get_audio_features(track_ids: List[str], tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Get detailed audio features for specific track IDs (from earlier results)."""
        feats = catalog.features(list(track_ids)[:50])
        session.log_call(tool_call_id, {"tool": "get_audio_features", "requested": len(track_ids),
                                        "results": len(feats)})
        return json.dumps(feats)

    return [search_catalog, get_user_taste, get_audio_features]
