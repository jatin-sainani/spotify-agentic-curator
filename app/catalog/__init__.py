"""Track catalogs (offline dataset index or live Spotify) behind one interface."""

from typing import Optional

from app.catalog.base import FEATURE_KEYS, CatalogProvider, FeatureFilters
from core.config import project_path, settings
from core.logging import logger


def get_catalog(kind: Optional[str] = None, profile: Optional[str] = None) -> CatalogProvider:
    """Create the configured catalog provider.

    Args:
        kind: "offline" or "spotify" (defaults to settings.catalog).
        profile: Offline listener profile name (defaults to settings.profile).
    """
    kind = (kind or settings.catalog).lower()
    profile = profile or settings.profile

    if kind == "offline":
        from app.catalog.offline import OfflineCatalog

        return OfflineCatalog(str(project_path(settings.catalog_path)), profile=profile)

    if kind == "spotify":
        from app.catalog.spotify import SpotifyCatalog
        from app.spotify.auth import SpotifyAuth
        from app.spotify.client import SpotifyClient

        fallback = None
        try:
            from app.catalog.offline import OfflineCatalog

            fallback = OfflineCatalog(str(project_path(settings.catalog_path)), profile=profile)
        except FileNotFoundError:
            logger.info("No offline index found; Spotify catalog will run without feature fallback")
        return SpotifyCatalog(SpotifyClient(SpotifyAuth()), feature_fallback=fallback)

    raise ValueError(f"Unknown catalog '{kind}' (expected 'offline' or 'spotify')")


__all__ = ["CatalogProvider", "FeatureFilters", "FEATURE_KEYS", "get_catalog"]
