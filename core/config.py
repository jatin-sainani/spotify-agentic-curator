"""Configuration management for spotify-agentic-curator."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, so paths work no matter where the process starts (an MCP client
# launches the server from its own working directory).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    """Resolve a configured path: absolute paths as-is, relative ones from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Spotify API credentials
    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    spotify_redirect_uri: Optional[str] = None

    # LLM provider settings.
    # llm_provider drives the agent (intent parsing, scoring, tool calling);
    # judge_provider drives the evaluation judge and falls back to llm_provider.
    llm_provider: str = "ollama"  # "ollama" | "anthropic"
    judge_provider: Optional[str] = None
    llm_api_key: Optional[str] = None  # legacy, unused

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Anthropic settings
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-5-5"

    # Catalog settings
    catalog: str = "offline"  # "offline" | "spotify"
    catalog_path: str = "data/catalog.parquet"
    profile: str = "default"  # offline taste profile name in profiles/

    # Application settings
    log_level: str = "INFO"
    debug: bool = False


# Global settings instance
settings = Settings()
