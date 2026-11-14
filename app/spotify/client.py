"""Spotify Web API client wrapper."""

from typing import Any, Dict, List

import httpx

from app.spotify.auth import SpotifyAuth
from core.logging import logger
from core.models import Track


class SpotifyClient:
    """Spotify Web API client with automatic token refresh."""

    BASE_URL = "https://api.spotify.com/v1"

    def __init__(self, auth: SpotifyAuth):
        """Initialize Spotify client with authentication."""
        self.auth = auth
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            timeout=30.0,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._client.close()

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers with valid access token."""
        token = self.auth.get_valid_access_token()
        if not token:
            raise ValueError("No valid access token available. Please run 'python -m app connect auth' first.")
        return {"Authorization": f"Bearer {token}"}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make authenticated request to Spotify API with automatic retry."""
        headers = self._get_auth_headers()
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers

        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"

        try:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.warning("Token expired, attempting refresh")
                # Try to refresh token and retry once
                if self.auth.refresh_access_token():
                    headers = self._get_auth_headers()
                    if "headers" in kwargs:
                        headers.update(kwargs["headers"])
                    kwargs["headers"] = headers
                    response = self._client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response.json()
                else:
                    raise ValueError("Authentication failed - please re-authenticate with 'python -m app connect auth'")
            elif e.response.status_code == 403:
                raise ValueError("Access forbidden - check your Spotify app permissions")
            elif e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "60")
                raise ValueError(f"Rate limited - try again after {retry_after} seconds")
            else:
                error_msg = f"Spotify API error: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except ValueError:  # error body wasn't JSON
                    pass
                raise ValueError(error_msg)
        except httpx.RequestError as e:
            raise ValueError(f"Request failed: {e}")

    def get_me(self) -> Dict[str, Any]:
        """Get current user profile."""
        logger.debug("Fetching user profile")
        try:
            return self._make_request("GET", "me")
        except ValueError as e:
            raise ValueError(f"Failed to get user profile: {e}")

    def get_top_tracks(
        self,
        limit: int = 5,
        time_range: str = "long_term"
    ) -> List[Track]:
        """Get user's top tracks.

        Args:
            limit: Maximum number of tracks to return (1-50)
            time_range: Over what time frame - short_term, medium_term, long_term

        Returns:
            List of Track objects
        """
        logger.debug(f"Fetching top tracks: limit={limit}, time_range={time_range}")

        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")

        if time_range not in ["short_term", "medium_term", "long_term"]:
            raise ValueError("time_range must be one of: short_term, medium_term, long_term")

        try:
            params = {
                "limit": limit,
                "time_range": time_range,
            }

            response = self._make_request("GET", "me/top/tracks", params=params)

            tracks = []
            for item in response.get("items", []):
                tracks.append(self._parse_track(item))

            return tracks

        except ValueError as e:
            raise ValueError(f"Failed to get top tracks: {e}")

    def get_recently_played_tracks(self, limit: int = 50) -> List[Track]:
        """Get user's recently played tracks.

        Args:
            limit: Maximum number of tracks to return (1-50)

        Returns:
            List of Track objects
        """
        logger.debug(f"Fetching recently played tracks: limit={limit}")

        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")

        try:
            params = {"limit": limit}

            response = self._make_request("GET", "me/player/recently-played", params=params)

            tracks = []
            for item in response.get("items", []):
                track_data = item.get("track")
                if track_data:
                    tracks.append(self._parse_track(track_data))

            return tracks

        except ValueError as e:
            raise ValueError(f"Failed to get recently played tracks: {e}")

    def get_saved_tracks(self, limit: int = 50) -> List[Track]:
        """Get user's saved/liked tracks.

        Args:
            limit: Maximum number of tracks to return (1-50)

        Returns:
            List of Track objects
        """
        logger.debug(f"Fetching saved tracks: limit={limit}")

        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")

        try:
            params = {"limit": limit}

            response = self._make_request("GET", "me/tracks", params=params)

            tracks = []
            for item in response.get("items", []):
                track_data = item.get("track")
                if track_data:
                    tracks.append(self._parse_track(track_data))

            return tracks

        except ValueError as e:
            raise ValueError(f"Failed to get saved tracks: {e}")

    def get_audio_features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get audio features for multiple tracks.

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            Dictionary mapping track IDs to audio features dictionaries
        """
        logger.debug(f"Fetching audio features for {len(track_ids)} tracks")

        if not track_ids:
            return {}

        # Spotify allows up to 100 tracks per request
        if len(track_ids) > 100:
            track_ids = track_ids[:100]

        try:
            ids_param = ",".join(track_ids)
            params = {"ids": ids_param}

            response = self._make_request("GET", "audio-features", params=params)

            features = {}
            for feature_data in response.get("audio_features", []):
                if feature_data:
                    track_id = feature_data["id"]
                    features[track_id] = feature_data

            return features

        except ValueError as e:
            raise ValueError(f"Failed to get audio features: {e}")

    def get_user_playlists(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's playlists.

        Args:
            limit: Maximum number of playlists to return (1-50)

        Returns:
            List of playlist objects with metadata
        """
        logger.debug(f"Fetching user playlists: limit={limit}")

        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")

        try:
            params = {"limit": limit}
            response = self._make_request("GET", "me/playlists", params=params)

            return response.get("items", [])

        except ValueError as e:
            logger.warning(f"Failed to get user playlists: {e}")
            return []

    def get_playlist_tracks(self, playlist_id: str, limit: int = 100) -> List[Track]:
        """Get tracks from a specific playlist.

        Args:
            playlist_id: Spotify playlist ID
            limit: Maximum number of tracks to return (1-100)

        Returns:
            List of Track objects
        """
        logger.debug(f"Fetching playlist tracks: playlist_id={playlist_id}, limit={limit}")

        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        try:
            params = {"limit": limit}
            response = self._make_request("GET", f"playlists/{playlist_id}/tracks", params=params)

            tracks = []
            for item in response.get("items", []):
                track_data = item.get("track")
                if track_data and track_data.get("id"):  # Skip local tracks or deleted tracks
                    tracks.append(self._parse_track(track_data))

            return tracks

        except ValueError as e:
            logger.warning(f"Failed to get playlist tracks for {playlist_id}: {e}")
            return []

    def search_tracks(
        self,
        query: str,
        limit: int = 50,
        market: str = "US",
        include_audio_features: bool = False,
    ) -> List[Track]:
        """Search for tracks using Spotify search API.

        Args:
            query: Search query (can include genre:, artist:, track:, etc.)
            limit: Maximum number of tracks to return (1-50)
            market: Market/country code (default: US)
            include_audio_features: Deprecated, ignored. Features are attached by
                SpotifyCatalog (with an offline fallback); fetching them here
                wasted a call that returns 403 for new apps and, being inside
                this try block, turned every search into an empty result.

        Returns:
            List of Track objects
        """
        logger.debug(f"Searching tracks: query='{query}', limit={limit}")

        if limit < 1 or limit > 50:
            raise ValueError("Limit must be between 1 and 50")

        try:
            params = {
                "q": query,
                "type": "track",
                "limit": limit,
                "market": market
            }

            response = self._make_request("GET", "search", params=params)

            return [self._parse_track(item) for item in response.get("tracks", {}).get("items", [])]

        except ValueError as e:
            logger.warning(f"Failed to search tracks with query '{query}': {e}")
            return []

    def get_track(self, track_id: str, market: str = "US") -> Track:
        """Get one track's metadata (bulk /tracks was removed for Dev Mode apps in Feb 2026)."""
        return self._parse_track(self._make_request("GET", f"tracks/{track_id}", params={"market": market}))

    # Playlist writes use the endpoints introduced by the February 2026 Web API
    # migration (POST /me/playlists and /playlists/{id}/items, replacing
    # /users/{id}/playlists and /playlists/{id}/tracks).

    def create_playlist(self, name: str, description: str = "", public: bool = False) -> Dict[str, Any]:
        """Create a playlist for the current user. Returns the playlist object."""
        logger.info(f"Creating Spotify playlist '{name}' (public={public})")
        return self._make_request(
            "POST", "me/playlists",
            json={"name": name, "description": description, "public": public},
        )

    def add_items_to_playlist(self, playlist_id: str, uris: List[str]) -> None:
        """Append track URIs to a playlist, 100 per request (API maximum)."""
        for i in range(0, len(uris), 100):
            self._make_request("POST", f"playlists/{playlist_id}/items", json={"uris": uris[i:i + 100]})
        logger.info(f"Added {len(uris)} tracks to playlist {playlist_id}")

    def _parse_track(self, track_data: Dict[str, Any]) -> Track:
        """Parse Spotify track data into Track model."""
        return Track(
            id=track_data["id"],
            name=track_data["name"],
            artist=track_data["artists"][0]["name"] if track_data["artists"] else "",
            album=track_data["album"]["name"] if track_data["album"] else "",
            duration_ms=track_data["duration_ms"],
            popularity=track_data.get("popularity"),
            external_url=track_data["external_urls"].get("spotify"),
            preview_url=track_data.get("preview_url"),
        )
