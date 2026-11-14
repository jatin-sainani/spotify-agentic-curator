"""Track ranking and selection for playlist curation.

Two-stage scoring:
  Stage 1 (heuristic): audio-feature fit to the intent + taste similarity.
  Stage 2 (LLM):       batch semantic relevance scoring of the stage-1 top-K.
Then diversity-aware selection (max tracks per artist).

Weights are explicit (RankWeights) so evaluation runs can ablate them.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from app.agent.intent import IntentNormalized
from app.core.draft_models import TrackRef
from core.logging import logger

MAX_PER_ARTIST = 3
STAGE2_TOP_K = 30
STAGE2_BATCH_SIZE = 10

# Distance outside a range at which a feature's fit decays to zero.
FEATURE_TOLERANCE = {"energy": 0.25, "valence": 0.25, "tempo": 30.0, "danceability": 0.25, "year": 4.0}

# " - 2015 Remaster", " - Live at ...", " - 12\" Version": everything after " - ".
_VERSION_SUFFIX = re.compile(r"\s+-\s+.*$")
# Parenthesized version markers: "(feat. X)", "(Remastered)", "[Live]", "(Radio Edit)".
_VERSION_PAREN = re.compile(
    r"\s*[\(\[][^\)\]]*\b(feat|ft|with|remaster\w*|live|version|edit|mix|remix|mono|stereo|"
    r"acoustic|demo|radio|deluxe|bonus|instrumental)\b[^\)\]]*[\)\]]"
)
_NON_WORD = re.compile(r"[^a-z0-9]+")


def song_key(track: TrackRef) -> str:
    """Identity of a *song* (not a recording): normalized title + primary artist.

    Catalogs list the same song many times (remasters, 12" versions, live cuts,
    "Blue Monday '88" vs "Blue Monday ('88)"); a playlist should contain it once.
    """
    title = _VERSION_PAREN.sub("", _VERSION_SUFFIX.sub("", track.name.lower()))
    artist = track.artists[0].lower() if track.artists else ""
    return f"{_NON_WORD.sub('', title)}|{_NON_WORD.sub('', artist)}"


Scored = Tuple[TrackRef, float]
SemanticScored = Tuple[TrackRef, float, Dict[str, Any]]


@dataclass(frozen=True)
class RankWeights:
    """Scoring weights. Defaults reproduce the original pipeline (the eval baseline)."""

    # Stage 1 (sum to 1.0)
    fit: float = 0.4          # audio-feature fit to the parsed intent
    taste: float = 0.4        # primary artist is one of the listener's top artists
    base: float = 0.2         # constant prior
    # Stage 2 (sum to 1.0)
    relevance: float = 0.7    # LLM relevance
    stage1: float = 0.2       # carry-over of the stage-1 score
    prior: float = 0.1        # constant prior (x 0.5)
    # Adaptive taste: if set, requests that don't ask for personal taste use this
    # taste weight instead, and the difference moves to feature fit.
    taste_when_impersonal: Optional[float] = None

    def stage1_weights(self, personal: bool) -> Tuple[float, float]:
        """(fit, taste) weights for a request."""
        if self.taste_when_impersonal is None or personal:
            return self.fit, self.taste
        return self.fit + self.taste - self.taste_when_impersonal, self.taste_when_impersonal


DEFAULT_WEIGHTS = RankWeights()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rank_and_select(
    candidates: List[TrackRef],
    intent: IntentNormalized,
    length: int,
    top_artists: Set[str],
    use_llm: Optional[bool] = None,
    weights: RankWeights = DEFAULT_WEIGHTS,
    request_text: Optional[str] = None,
) -> List[TrackRef]:
    """Rank candidates and select a diverse playlist.

    Args:
        candidates: Candidate tracks.
        intent: Normalized intent with constraints.
        length: Number of tracks to select.
        top_artists: Listener's top artists (taste similarity).
        use_llm: Force LLM scoring on/off; None = use it if the provider is reachable.
        weights: Scoring weights.
        request_text: The user's own words, shown to the LLM scorer alongside the
            parsed description (paraphrases drop nuance and feedback).

    Returns:
        Selected tracks with populated 'why' fields.
    """
    logger.info(f"Ranking {len(candidates)} candidates for {length} slots ({intent.description})")
    if not candidates:
        logger.warning("No candidates provided for ranking")
        return []

    from app.agent.safety import filter_flagged

    stage1 = _stage1_pre_score(filter_flagged(candidates, "ranking"), intent, top_artists, weights)
    top = stage1[:STAGE2_TOP_K]

    if use_llm is None:
        use_llm = _llm_available()

    if use_llm:
        scored = _stage2_llm_score(top, intent, weights=weights, request_text=request_text)
    else:
        logger.info("Stage 2 skipped (LLM unavailable or disabled); using stage-1 scores")
        scored = heuristic_only(stage1)

    selected = select_with_diversity(scored, length)
    for track, final_score, semantic in selected:
        track.why = _generate_semantic_selection_reason(track, final_score, semantic, intent, top_artists)
    logger.info(f"Selected {len(selected)}/{length} tracks")
    return [track for track, _, _ in selected]


def heuristic_only(stage1: List[Scored]) -> List[SemanticScored]:
    """Wrap stage-1 scores in the stage-2 shape when no LLM scoring ran."""
    return [(t, s, {"relevance": None, "mood_tags": [], "why": ""}) for t, s in stage1]


def select_with_diversity(
    scored: List[SemanticScored],
    length: int,
    max_per_artist: int = MAX_PER_ARTIST,
) -> List[SemanticScored]:
    """Pick the highest-scoring tracks, at most max_per_artist per primary artist.

    The artist cap is soft: if honoring it would leave the playlist short, the
    best skipped tracks fill the remaining slots (the requested length wins).
    Duplicate songs (same title + artist, different recording) are never repeated.
    Returned in score order.
    """
    ordered = sorted(scored, key=lambda x: x[1], reverse=True)
    picked: List[SemanticScored] = []
    skipped: List[SemanticScored] = []
    seen_songs: Set[str] = set()
    per_artist: Counter = Counter()
    for item in ordered:
        if len(picked) >= length:
            break
        track = item[0]
        key = song_key(track)
        if key in seen_songs:
            continue
        seen_songs.add(key)
        artist = track.artists[0] if track.artists else ""
        if artist and per_artist[artist] >= max_per_artist:
            skipped.append(item)
            continue
        picked.append(item)
        if artist:
            per_artist[artist] += 1

    if len(picked) < length and skipped:
        picked.extend(skipped[:length - len(picked)])
        picked.sort(key=lambda x: x[1], reverse=True)
    return picked


# ---------------------------------------------------------------------------
# Stage 1: heuristic
# ---------------------------------------------------------------------------

def _stage1_pre_score(
    candidates: List[TrackRef],
    intent: IntentNormalized,
    top_artists: Set[str],
    weights: RankWeights = DEFAULT_WEIGHTS,
) -> List[Scored]:
    """Score by feature fit and taste. Returns (track, score) sorted descending."""
    fit_w, taste_w = weights.stage1_weights(bool((intent.structure or {}).get("personal")))
    scored = []
    for track in candidates:
        taste = 1.0 if (track.artists and track.artists[0] in top_artists) else 0.0
        fit = _calculate_constraint_fit(track, intent.constraints) if track.audio_features else 0.5
        score = fit_w * fit + taste_w * taste + weights.base
        scored.append((track, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _range_fit(value: float, lo: float, hi: float, tolerance: float) -> float:
    """1.0 inside [lo, hi]; decays linearly to 0 at `tolerance` outside the range."""
    if lo <= value <= hi:
        return 1.0
    distance = lo - value if value < lo else value - hi
    return max(0.0, 1.0 - distance / tolerance)


def _calculate_constraint_fit(track: TrackRef, constraints: Dict[str, Any]) -> float:
    """How well a track's audio features fit the intent's ranges (0.0-1.0).

    Averages per-feature fit over the constraints the track has features for.
    Tracks without features score 0.0 (callers decide how to treat unknowns).
    """
    features = track.audio_features
    if not features:
        return 0.0

    fits = []
    year_range = constraints.get("year_range")
    if year_range and track.year:
        fits.append(_range_fit(float(track.year), float(year_range[0]), float(year_range[1]), FEATURE_TOLERANCE["year"]))
    for key, feat in (("energy_range", "energy"), ("valence_range", "valence"), ("bpm_range", "tempo")):
        rng = constraints.get(key)
        if rng and feat in features:
            fits.append(_range_fit(float(features[feat]), float(rng[0]), float(rng[1]), FEATURE_TOLERANCE[feat]))

    dance_min = constraints.get("danceability_min")
    if dance_min is not None and "danceability" in features:
        fits.append(_range_fit(float(features["danceability"]), float(dance_min), 1.0, FEATURE_TOLERANCE["danceability"]))

    return sum(fits) / len(fits) if fits else 0.0


def _calculate_diversity_penalty(track: TrackRef, selected_artists: Counter) -> float:
    """Diversity multiplier (1.0 = new artist, decreasing with repeats)."""
    if not track.artists:
        return 0.8
    count = selected_artists.get(track.artists[0], 0)
    return {0: 1.0, 1: 0.8, 2: 0.5}.get(count, 0.2)


# ---------------------------------------------------------------------------
# Stage 2: LLM semantic scoring
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    try:
        from app.agent.llm import provider_available

        return provider_available()
    except ImportError as e:
        logger.warning(f"LLM libraries unavailable: {e}")
        return False


def _stage2_llm_score(
    candidates: List[Scored],
    intent: IntentNormalized,
    batch_size: int = STAGE2_BATCH_SIZE,
    weights: RankWeights = DEFAULT_WEIGHTS,
    request_text: Optional[str] = None,
) -> List[SemanticScored]:
    """Score candidates with the LLM in batches and blend with stage-1 scores.

    A failed batch falls back to neutral relevance (50) for its tracks only.
    """
    from app.agent.mood_priors import get_relevant_priors

    request = scoring_request(intent, request_text)
    priors_context = _priors_context(get_relevant_priors(request))
    results: List[SemanticScored] = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        tracks = [t for t, _ in batch]
        try:
            semantic = _llm_batch_score(tracks, request, priors_context)
        except Exception as e:
            logger.warning(f"LLM scoring failed for batch {i // batch_size}: {e}")
            semantic = [_default_semantic() for _ in tracks]

        for (track, s1), sem in zip(batch, semantic):
            relevance = sem["relevance"] / 100.0
            final = weights.relevance * relevance + weights.stage1 * s1 + weights.prior * 0.5
            results.append((track, final, sem))

    logger.info(f"Stage 2 scored {len(results)} tracks in {-(-len(candidates) // batch_size)} LLM calls")
    return results


# Backward-compatible name used by graph.py / commands.py.
_stage2_ollama_score = _stage2_llm_score


def scoring_request(intent: IntentNormalized, request_text: Optional[str] = None) -> str:
    """What the LLM scores against: the user's words first, the parse second."""
    if not request_text:
        return intent.description
    if intent.description and intent.description.lower() not in request_text.lower():
        return f"{request_text} (interpreted as: {intent.description})"
    return request_text


def format_track_line(n: int, track: TrackRef) -> str:
    """One numbered, compact line of evidence about a track for the LLM."""
    from app.agent.safety import for_prompt

    track = for_prompt(track)
    parts = [f"{n}. \"{track.name}\" by {', '.join(track.artists) or 'Unknown'}"]
    # genres[0] is the catalog's coarse bucket (often misleading: New Order is filed
    # under "Rock"); lead with the specific genres so the model doesn't anchor on it.
    specific, bucket = track.genres[1:4], track.genres[:1]
    if specific:
        parts.append(f"genres: {', '.join(specific)} (catalog bucket: {bucket[0]})")
    elif bucket:
        parts.append(f"genre: {bucket[0]}")
    if track.year:
        parts.append(f"year: {track.year}")
    f = track.audio_features or {}
    feats = [f"{k}={f[k]:.2f}" for k in ("energy", "valence", "acousticness", "danceability") if k in f]
    if "tempo" in f:
        feats.append(f"tempo={f['tempo']:.0f}")
    if feats:
        parts.append(" ".join(feats))
    return " | ".join(parts)


def _llm_batch_score(
    tracks: List[TrackRef],
    request: str,
    priors_context: str,
) -> List[Dict[str, Any]]:
    """One LLM call for a batch. Returns normalized semantic dicts aligned to `tracks`."""
    from app.agent.llm import invoke_score_chain

    block = "\n".join(format_track_line(n, t) for n, t in enumerate(tracks, 1))
    raw = invoke_score_chain(block, request, priors_context)
    items = raw.get("scores", []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []

    by_n: Dict[int, Dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict):
            try:
                by_n[int(item.get("n"))] = item
            except (TypeError, ValueError):
                continue

    missing = [n for n in range(1, len(tracks) + 1) if n not in by_n]
    if missing:
        logger.warning(f"LLM returned {len(by_n)}/{len(tracks)} scores; defaulting tracks {missing}")
    return [_normalize_semantic(by_n.get(n)) for n in range(1, len(tracks) + 1)]


def _default_semantic() -> Dict[str, Any]:
    return {"relevance": 50, "mood_tags": [], "why": ""}


def _normalize_semantic(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return _default_semantic()
    try:
        relevance = max(0, min(100, int(float(data.get("relevance", 50)))))
    except (TypeError, ValueError):
        relevance = 50
    tags = data.get("mood_tags", [])
    tags = [str(t) for t in tags][:8] if isinstance(tags, list) else []
    why = str(data.get("why") or "")[:120]
    return {"relevance": relevance, "mood_tags": tags, "why": why}


def _priors_context(priors: Optional[Dict[str, Any]]) -> str:
    if not priors:
        return ""
    genres = priors.get("top_genres", [])[:10]
    keywords = priors.get("top_lyric_keywords", [])[:15]
    ranges = priors.get("feature_ranges", {})
    return (
        f"Mood priors - typical genres: {', '.join(genres)}; "
        f"common lyric words: {', '.join(keywords)}; "
        f"typical feature ranges: {', '.join(f'{k}: {v}' for k, v in ranges.items())}"
    )


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _feature_notes(track: TrackRef, constraints: Dict[str, Any]) -> List[str]:
    f = track.audio_features or {}
    notes = []
    checks = (("energy_range", "energy", "energy level fits"),
              ("valence_range", "valence", "mood matches"),
              ("bpm_range", "tempo", "tempo fits"))
    for key, feat, label in checks:
        rng = constraints.get(key)
        if rng and feat in f and rng[0] <= f[feat] <= rng[1]:
            notes.append(label)
    dance_min = constraints.get("danceability_min")
    if dance_min and "danceability" in f and f["danceability"] >= dance_min:
        notes.append("danceable enough")
    year_range = constraints.get("year_range")
    if year_range and track.year and year_range[0] <= track.year <= year_range[1]:
        notes.append(f"from the {str(year_range[0])[2:]}s" if year_range[1] - year_range[0] == 9 else "right era")
    return notes


def _generate_semantic_selection_reason(
    track: TrackRef,
    final_score: float,
    semantic_data: Dict[str, Any],
    intent: Optional[IntentNormalized] = None,
    top_artists: Optional[Set[str]] = None,
) -> str:
    """Explain a selection from LLM output plus feature and taste evidence."""
    parts = []
    relevance = semantic_data.get("relevance")
    if relevance is None:
        parts.append("excellent overall match" if final_score > 0.7 else
                     "good overall match" if final_score > 0.5 else "fair overall match")
    elif relevance >= 80:
        parts.append("highly relevant")
    elif relevance >= 60:
        parts.append("well matched")
    else:
        parts.append("partial match")

    if semantic_data.get("mood_tags"):
        parts.append(f"mood: {', '.join(semantic_data['mood_tags'][:2])}")
    if intent is not None:
        parts.extend(_feature_notes(track, intent.constraints))
    if top_artists and track.artists and track.artists[0] in top_artists:
        parts.append("from your top artists")
    why = semantic_data.get("why")
    if why:
        parts.append(why)
    return "Selected because: " + "; ".join(parts)


def _generate_selection_reason(
    track: TrackRef,
    score: float,
    intent: IntentNormalized,
    top_artists: Set[str],
    relaxation_round: int,
) -> str:
    """Legacy heuristic-only explanation (kept for callers of the old API)."""
    reasons = ["excellent overall match" if score > 0.7 else "good overall match" if score > 0.5 else "fair overall match"]
    reasons.extend(_feature_notes(track, intent.constraints))
    if track.artists and track.artists[0] in top_artists:
        reasons.append("from your top artists")
    if relaxation_round > 0:
        reasons.append(f"selected after constraint relaxation (round {relaxation_round})")
    return "Selected because: " + ", ".join(reasons)
