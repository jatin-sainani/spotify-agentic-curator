"""Spotify OAuth Authorization Code flow with PKCE implementation."""

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import httpx

from core.config import PROJECT_ROOT, settings
from core.logging import logger


class SpotifyAuth:
    """Handles Spotify OAuth Authorization Code flow with PKCE."""

    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    CACHE_DIR = str(PROJECT_ROOT / ".cache")
    TOKEN_FILE = os.path.join(CACHE_DIR, "spotify_tokens.json")

    SCOPES = [
        "playlist-modify-private",
        "playlist-modify-public",
        "user-top-read",
        "user-read-recently-played",
        "user-library-read"
    ]

    def __init__(self):
        """Initialize Spotify OAuth handler."""
        self.client_id = settings.spotify_client_id
        self.redirect_uri = settings.spotify_redirect_uri or "http://localhost:8888/callback"

        if not self.client_id:
            raise ValueError("SPOTIFY_CLIENT_ID must be configured in environment")

        # Ensure cache directory exists
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and code challenge."""
        # Generate code verifier (43-128 characters)
        code_verifier = secrets.token_urlsafe(64)[:128]

        # Generate code challenge using S256
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')

        return code_verifier, code_challenge

    def get_authorization_url(self, state: str, code_challenge: str) -> str:
        """Generate Spotify authorization URL with PKCE."""
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "state": state,
            "scope": " ".join(self.SCOPES),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(self, code: str, code_verifier: str) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens using PKCE."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }

        try:
            response = httpx.post(self.TOKEN_URL, data=data)
            response.raise_for_status()
            token_data = response.json()

            # Add timestamp for token expiry tracking
            token_data["expires_at"] = time.time() + token_data["expires_in"]

            # Save tokens
            self._save_tokens(token_data)
            logger.info("Successfully exchanged code for tokens")
            return token_data

        except httpx.HTTPStatusError as e:
            error_msg = f"Token exchange failed: {e.response.status_code}"
            if e.response.status_code == 400:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('error_description', error_data.get('error', 'Unknown error'))}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Request error during token exchange: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def refresh_access_token(self) -> Optional[Dict[str, Any]]:
        """Refresh access token using refresh token."""
        tokens = self._load_tokens()
        if not tokens or "refresh_token" not in tokens:
            logger.warning("No refresh token available")
            return None

        data = {
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": self.client_id,
        }

        try:
            response = httpx.post(self.TOKEN_URL, data=data)
            response.raise_for_status()
            new_token_data = response.json()

            # Update tokens, preserving refresh token if not provided
            if "refresh_token" not in new_token_data:
                new_token_data["refresh_token"] = tokens["refresh_token"]

            new_token_data["expires_at"] = time.time() + new_token_data["expires_in"]
            self._save_tokens(new_token_data)
            logger.info("Successfully refreshed access token")
            return new_token_data

        except httpx.HTTPStatusError as e:
            error_msg = f"Token refresh failed: {e.response.status_code}"
            if e.response.status_code == 400:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('error_description', error_data.get('error', 'Unknown error'))}"
            logger.error(error_msg)
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error during token refresh: {e}")
            return None

    def get_valid_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        tokens = self._load_tokens()
        if not tokens:
            return None

        # Check if token is still valid (with 5 minute buffer)
        if time.time() < tokens.get("expires_at", 0) - 300:
            return tokens["access_token"]

        # Token expired or will expire soon, try to refresh
        logger.info("Access token expired, attempting refresh")
        refreshed = self.refresh_access_token()
        return refreshed["access_token"] if refreshed else None

    def authenticate(self) -> bool:
        """Perform full OAuth flow with PKCE if needed."""
        # Check if we already have valid tokens
        if self.get_valid_access_token():
            logger.info("Valid tokens already exist")
            return True

        # Generate PKCE pair
        code_verifier, code_challenge = self._generate_pkce_pair()

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(16)

        # Create authorization URL
        auth_url = self.get_authorization_url(state, code_challenge)

        # Start local server to handle callback
        callback_received = threading.Event()
        auth_result = {"success": False, "error": None}
        auth = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    parsed_path = urllib.parse.urlparse(self.path)
                    query_params = urllib.parse.parse_qs(parsed_path.query)

                    # Check state for CSRF protection
                    if query_params.get("state", [None])[0] != state:
                        auth_result["error"] = "State parameter mismatch - possible CSRF attack"
                        self.send_error(400, "Invalid state parameter")
                        callback_received.set()
                        return

                    if "error" in query_params:
                        error = query_params["error"][0]
                        error_desc = query_params.get("error_description", ["Unknown error"])[0]
                        auth_result["error"] = f"Authorization failed: {error} - {error_desc}"
                        self.send_error(400, f"Authorization failed: {error}")
                        callback_received.set()
                        return

                    if "code" not in query_params:
                        auth_result["error"] = "Missing authorization code in callback"
                        self.send_error(400, "Missing authorization code")
                        callback_received.set()
                        return

                    # Success - exchange code for tokens
                    auth_code = query_params["code"][0]
                    try:
                        tokens = auth.exchange_code_for_tokens(auth_code, code_verifier)
                        auth_result["success"] = True
                        auth_result["tokens"] = tokens

                        self.send_response(200)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"""
                        <html>
                        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                        <h1 style="color: #1DB954;">Authorization successful!</h1>
                        <p>You can close this window and return to the terminal.</p>
                        <p style="color: #666; font-size: 14px;">Spotify Agentic Curator is now connected.</p>
                        </body>
                        </html>
                        """)

                    except Exception as e:
                        auth_result["error"] = f"Token exchange failed: {str(e)}"
                        self.send_error(500, f"Token exchange failed: {str(e)}")

                    callback_received.set()

                except Exception as e:
                    auth_result["error"] = f"Callback processing failed: {str(e)}"
                    self.send_error(500, f"Internal error: {str(e)}")
                    callback_received.set()

            def log_message(self, format, *args):
                # Suppress default server logs
                pass

        # Extract port from redirect URI
        try:
            parsed_uri = urllib.parse.urlparse(self.redirect_uri)
            port = parsed_uri.port or (443 if parsed_uri.scheme == "https" else 80)
            host = parsed_uri.hostname or "localhost"
        except Exception:
            raise ValueError(f"Invalid redirect URI format: {self.redirect_uri}")

        try:
            server = HTTPServer((host, port), CallbackHandler)
            server.timeout = 1  # Allow checking for shutdown

            logger.info(f"Starting local server on {host}:{port}")

            # Open browser
            logger.info("Opening browser for Spotify authorization...")
            webbrowser.open(auth_url)

            # Wait for callback with timeout (5 minutes)
            timeout_time = time.time() + 300
            while not callback_received.is_set() and time.time() < timeout_time:
                server.handle_request()

            server.server_close()

            if not callback_received.is_set():
                logger.error("Authorization timeout - no callback received within 5 minutes")
                return False

            if auth_result.get("error"):
                logger.error(f"Authorization failed: {auth_result['error']}")
                return False

            if auth_result.get("success"):
                logger.info("Authorization completed successfully")
                return True
            else:
                logger.error("Authorization failed with unknown error")
                return False

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def _save_tokens(self, token_data: Dict[str, Any]):
        """Save tokens to cache file."""
        try:
            with open(self.TOKEN_FILE, "w") as f:
                json.dump(token_data, f, indent=2)
            logger.debug(f"Tokens saved to {self.TOKEN_FILE}")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def _load_tokens(self) -> Optional[Dict[str, Any]]:
        """Load tokens from cache file."""
        try:
            if os.path.exists(self.TOKEN_FILE):
                with open(self.TOKEN_FILE) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
        return None

    def clear_tokens(self):
        """Clear stored tokens."""
        try:
            if os.path.exists(self.TOKEN_FILE):
                os.remove(self.TOKEN_FILE)
                logger.info("Tokens cleared")
        except Exception as e:
            logger.error(f"Failed to clear tokens: {e}")
