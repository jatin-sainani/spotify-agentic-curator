"""Candidate track retrieval for playlist curation.

Builds a deduplicated candidate pool from any CatalogProvider:
~30% from the listener's taste and ~70% from intent-driven catalog search.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional

from app.agent.intent import IntentNormalized
from app.catalog.base import CatalogProvider, FeatureFilters
from app.core.draft_models import TrackRef
from core.logging import logger

TASTE_SHARE = 0.3
MAX_QUERIES = 4

# Phrases in a request that map to catalog genre labels. Multi-word and compound
# genres come first and are matched as whole words, so "synth-pop" yields
# "synthpop" (not "pop") and "new wave" isn't read as two unrelated words.
GENRE_HINTS = {
    "synth-pop": "synthpop", "synthpop": "synthpop", "synth pop": "synthpop",
    "new wave": "new wave", "trip hop": "trip hop", "drum and bass": "drum and bass",
    "bossa nova": "bossa nova", "k-pop": "k-pop", "j-pop": "j-pop", "pop punk": "pop punk",
    "dream pop": "dream pop", "neo soul": "neo soul", "new jack swing": "new jack swing",
    "yacht rock": "yacht rock", "outlaw country": "outlaw country", "hip-hop": "hip hop",
    "hip hop": "hip hop", "r&b": "r&b", "lo-fi": "lo-fi", "lofi": "lo-fi",
    "rock": "rock", "indie": "indie", "punk": "punk", "metal": "metal", "grunge": "grunge",
    "emo": "emo", "shoegaze": "shoegaze", "pop": "pop", "electronic": "electronic",
    "edm": "edm", "house": "house", "techno": "techno", "disco": "disco", "funk": "funk",
    "motown": "motown", "rap": "rap", "soul": "soul", "jazz": "jazz", "blues": "blues",
    "folk": "folk", "acoustic": "acoustic", "country": "country", "bluegrass": "bluegrass",
    "classical": "classical", "piano": "piano", "ambient": "ambient", "reggae": "reggae",
    "latin": "latin", "afrobeat": "afrobeat", "celtic": "celtic",
}


def _genre_hints(text: str) -> List[str]:
    """Genre labels named in the text, longest phrases first, without overlaps."""
    hits, taken = [], []
    for phrase in sorted(GENRE_HINTS, key=len, reverse=True):
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text):
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue  # part of a longer phrase already matched
            taken.append((m.start(), m.end()))
            hits.append((m.start(), GENRE_HINTS[phrase]))
    found: List[str] = []
    for _, genre in sorted(hits):  # in the order the request mentions them
        if genre not in found:
            found.append(genre)
    return found


def build_candidates(
    catalog: CatalogProvider,
    limit: int = 200,
    intent: Optional[IntentNormalized] = None,
    intent_text: Optional[str] = None,
    taste_share: float = TASTE_SHARE,
) -> List[TrackRef]:
    """Build a pool of candidate tracks.

    Args:
        catalog: Track source (offline index or live Spotify).
        limit: Maximum number of candidates to return.
        intent: Parsed intent (drives search queries and feature filters).
        intent_text: The user's raw request, used for extra search terms.
        taste_share: Fraction of the pool reserved for the listener's taste tracks
            (ignored without an intent, where the pool is all taste).

    Returns:
        Deduplicated TrackRefs, audio features attached where available.
    """
    logger.info(f"Building candidate pool from '{catalog.name}' catalog (limit: {limit})")

    candidates: Dict[str, TrackRef] = {}
    sources: Dict[str, int] = defaultdict(int)

    taste_limit = int(limit * taste_share) if intent else limit

    try:
        for ref in (catalog.get_user_taste(limit=taste_limit) if taste_limit > 0 else []):
            if ref.id not in candidates:
                candidates[ref.id] = ref
                sources["user_taste"] += 1
    except Exception as e:
        logger.warning(f"Failed to fetch user taste tracks: {e}")

    if intent:
        _add_intent_matches(catalog, candidates, sources, intent, intent_text, limit)

    for source, count in sources.items():
        logger.info(f"Fetched {count} tracks from {source}")

    pool = list(candidates.values())[:limit]
    with_features = sum(1 for r in pool if r.audio_features)
    logger.info(f"Candidate pool: {len(pool)} unique tracks ({with_features} with audio features)")
    return pool


def _add_intent_matches(
    catalog: CatalogProvider,
    candidates: Dict[str, TrackRef],
    sources: Dict[str, int],
    intent: IntentNormalized,
    intent_text: Optional[str],
    limit: int,
) -> None:
    """Search the catalog for the intent, relaxing feature filters if results are thin."""
    queries = build_search_queries(intent, intent_text)
    filters = FeatureFilters.from_constraints(intent.constraints)
    per_query = max(10, (limit - len(candidates)) // max(1, len(queries)) + 5)

    # Pass 1 uses every filter; pass 2 relaxes audio features but keeps the era,
    # since "80s" in a request is a hard requirement, not a mood hint.
    relaxed = filters.era_only()
    for use_filters in (True, False):
        if len(candidates) >= limit:
            break
        if not use_filters and dict(relaxed) == dict(filters):
            break  # second pass would be identical
        source = "intent_search" if use_filters else "intent_search_relaxed"
        for query in queries:
            if len(candidates) >= limit:
                break
            try:
                results = catalog.search(query, limit=per_query, filters=filters if use_filters else (relaxed or None))
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
                continue
            for ref in results:
                if ref.id not in candidates:
                    candidates[ref.id] = ref
                    sources[source] += 1
                    if len(candidates) >= limit:
                        break


def build_search_queries(intent: IntentNormalized, intent_text: Optional[str] = None) -> List[str]:
    """Turn an intent into a few plain-text catalog queries.

    Plain words (no Spotify field syntax) so they work for both the offline
    index and Spotify search.
    """
    text = f"{intent_text or ''} {intent.description}".lower()
    queries: List[str] = []
    queries.extend(_genre_hints(text)[:2])

    if intent_text:
        queries.append(intent_text)
    if intent.description and intent.description.lower() != (intent_text or "").lower():
        queries.append(intent.description)

    # Dedupe while preserving order.
    seen, out = set(), []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:MAX_QUERIES] or [intent.description or "music"]
