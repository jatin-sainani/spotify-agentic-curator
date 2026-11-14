"""Catalog abstraction: where candidate tracks come from.

The agent never talks to Spotify (or a local dataset) directly. It talks to a
CatalogProvider, so the same pipeline runs against:

- OfflineCatalog: a local index built from a songs dataset (no API keys,
  reproducible, has audio features that the Spotify Web API no longer serves)
- SpotifyCatalog: the live Spotify Web API for the authenticated user
"""

import re
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from app.core.draft_models import TrackRef

FEATURE_KEYS = (
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
)


_DECADE_WORDS = {"fifties": 1950, "sixties": 1960, "seventies": 1970, "eighties": 1980, "nineties": 1990}
_DECADE_RE = re.compile(r"(?<![\w])'?((?:19|20)?\d0)'?s\b")


def parse_era(text: str) -> Optional[Tuple[int, int]]:
    """Release-year range implied by decade words: "80s", "1980s", "'90s", "2000s", "sixties".

    Multiple decades ("70s and 80s") yield their combined span. Returns None if
    the text names no decade.
    """
    text = (text or "").lower()
    starts = []
    for match in _DECADE_RE.finditer(text):
        token = match.group(1)
        if len(token) == 2:  # "80s" -> 1980, "00s" -> 2000, "10s" -> 2010, "20s" -> 2020
            year = int(token)
            starts.append(1900 + year if year >= 30 else 2000 + year)
        else:
            starts.append(int(token))
    starts += [y for w, y in _DECADE_WORDS.items() if w in text]
    starts = [y for y in starts if 1900 <= y <= 2020]
    if not starts:
        return None
    return min(starts), max(starts) + 9


class FeatureFilters(dict):
    """Feature range filters, in the normalized intent constraint format.

    Keys (all optional): energy_range [lo, hi], valence_range [lo, hi],
    bpm_range [lo, hi], danceability_min float, year_range [first, last].
    """

    @classmethod
    def from_constraints(cls, constraints: Optional[Dict[str, Any]]) -> "FeatureFilters":
        constraints = constraints or {}
        out = cls()
        for key in ("energy_range", "valence_range", "bpm_range"):
            rng = constraints.get(key)
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                out[key] = [float(rng[0]), float(rng[1])]
        if constraints.get("danceability_min") is not None:
            out["danceability_min"] = float(constraints["danceability_min"])
        rng = constraints.get("year_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            out["year_range"] = [int(rng[0]), int(rng[1])]
        return out

    def era_only(self) -> "FeatureFilters":
        """Just the era filter (kept when audio-feature filters are relaxed)."""
        return FeatureFilters({k: v for k, v in self.items() if k == "year_range"})

    def is_unconstrained(self) -> bool:
        defaults = {
            "energy_range": [0.0, 1.0],
            "valence_range": [0.0, 1.0],
        }
        for key, value in self.items():
            if key in defaults and value == defaults[key]:
                continue
            if key == "bpm_range" and value[0] <= 60 and value[1] >= 180:
                continue
            return False
        return True


@runtime_checkable
class CatalogProvider(Protocol):
    """Interface every track source implements."""

    name: str

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: Optional[FeatureFilters] = None,
    ) -> List[TrackRef]:
        """Free-text search (genre, artist, title, mood words), optionally feature-filtered."""
        ...

    def get_tracks(self, track_ids: List[str]) -> List[TrackRef]:
        """Look up tracks by ID (unknown IDs are skipped)."""
        ...

    def features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Audio features keyed by track ID (unknown IDs are omitted)."""
        ...

    def get_user_taste(self, limit: int = 50) -> List[TrackRef]:
        """Tracks representing the listener's taste."""
        ...

    def top_artists(self) -> Set[str]:
        """Artist names the listener is known to like."""
        ...
