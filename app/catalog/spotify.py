"""Live Spotify catalog adapter.

Wraps the existing SpotifyClient behind the CatalogProvider interface. Spotify
no longer serves /audio-features to new apps, so features fall back to the
offline index (looked up by track ID) when the API refuses them.
"""

from typing import Any, Dict, List, Optional, Set

from app.catalog.base import FeatureFilters
from app.core.draft_models import TrackRef
from app.spotify.client import SpotifyClient
from core.logging import logger
from core.models import Track


class SpotifyCatalog:
    """CatalogProvider over the Spotify Web API for the authenticated user."""

    name = "spotify"

    def __init__(self, client: SpotifyClient, feature_fallback: Optional[Any] = None):
        """
        Args:
            client: Authenticated SpotifyClient.
            feature_fallback: Optional provider with a .features(ids) method
                (e.g. OfflineCatalog) used when the API returns no features.
        """
        self.client = client
        self.feature_fallback = feature_fallback
        self._top_artists: Optional[Set[str]] = None

    @staticmethod
    def _to_ref(track: Track) -> TrackRef:
        return TrackRef(
            uri=f"spotify:track:{track.id}",
            id=track.id,
            name=track.name,
            artists=[track.artist] if track.artist else [],
            popularity=track.popularity,
        )

    def _attach_features(self, refs: List[TrackRef]) -> List[TrackRef]:
        feats = self.features([r.id for r in refs])
        for ref in refs:
            ref.audio_features = feats.get(ref.id)
        return refs

    def search(self, query: str, limit: int = 20, filters: Optional[FeatureFilters] = None) -> List[TrackRef]:
        tracks = self.client.search_tracks(query=query, limit=min(50, limit), include_audio_features=False)
        refs = self._attach_features([self._to_ref(t) for t in tracks])
        if filters and not filters.is_unconstrained():
            refs = [r for r in refs if _passes(r, filters)]
        return refs[:limit]

    def get_tracks(self, track_ids: List[str]) -> List[TrackRef]:
        # Dev Mode apps lost bulk /tracks in Feb 2026: use the offline index where it
        # knows the track, and single-track lookups for the rest.
        found: Dict[str, TrackRef] = {}
        if self.feature_fallback is not None and hasattr(self.feature_fallback, "get_tracks"):
            found = {r.id: r for r in self.feature_fallback.get_tracks(track_ids)}
        missing = [tid for tid in track_ids if tid not in found]
        for tid in missing:
            try:
                found[tid] = self._to_ref(self.client.get_track(tid))
            except Exception as e:
                logger.warning(f"Could not look up track {tid}: {e}")
        self._attach_features([found[t] for t in missing if t in found])
        return [found[t] for t in track_ids if t in found]

    def features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        feats: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            try:
                feats.update(self.client.get_audio_features(batch) or {})
            except Exception as e:  # 403 for apps without audio-features access
                logger.debug(f"Spotify audio-features unavailable: {e}")
        missing = [tid for tid in track_ids if tid not in feats]
        if missing and self.feature_fallback is not None:
            feats.update(self.feature_fallback.features(missing))
        return feats

    def get_user_taste(self, limit: int = 50) -> List[TrackRef]:
        seen: Dict[str, Track] = {}
        for time_range in ("short_term", "medium_term", "long_term"):
            try:
                for t in self.client.get_top_tracks(limit=min(50, limit), time_range=time_range):
                    seen.setdefault(t.id, t)
            except Exception as e:
                logger.warning(f"Failed to fetch {time_range} top tracks: {e}")
            if len(seen) >= limit:
                break
        if len(seen) < limit:
            try:
                for t in self.client.get_saved_tracks(limit=min(50, limit - len(seen))):
                    seen.setdefault(t.id, t)
            except Exception as e:
                logger.warning(f"Failed to fetch saved tracks: {e}")
        refs = [self._to_ref(t) for t in list(seen.values())[:limit]]
        return self._attach_features(refs)

    def top_artists(self) -> Set[str]:
        if self._top_artists is None:
            try:
                tracks = self.client.get_top_tracks(limit=50, time_range="medium_term")
                self._top_artists = {t.artist for t in tracks if t.artist}
            except Exception as e:
                logger.warning(f"Failed to fetch top artists: {e}")
                self._top_artists = set()
        return self._top_artists


def _passes(ref: TrackRef, filters: FeatureFilters) -> bool:
    f = ref.audio_features
    if not f:
        return True  # unknown features: don't exclude, let ranking decide
    checks = (("energy_range", "energy"), ("valence_range", "valence"), ("bpm_range", "tempo"))
    for key, feat in checks:
        if key in filters and feat in f:
            lo, hi = filters[key]
            if not lo <= f[feat] <= hi:
                return False
    if filters.get("danceability_min") is not None and "danceability" in f:
        if f["danceability"] < filters["danceability_min"]:
            return False
    return True
