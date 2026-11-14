"""Intent parsing for playlist curation with normalized schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.logging import logger


class IntentNormalized(BaseModel):
    """Normalized intent schema for downstream ranking."""

    mode: Literal["curate", "top_tracks"] = Field(..., description="Curation mode")
    description: str = Field(..., description="Human-readable description")
    constraints: dict = Field(..., description="Audio feature constraints with ranges")
    structure: dict = Field(..., description="Playlist structure with phases")

    model_config = ConfigDict(populate_by_name=True)


def parse_intent(intent_text: str, length: int) -> IntentNormalized:
    """Parse intent text into normalized schema for downstream ranking.

    Args:
        intent_text: User intent/prompt text
        length: Target playlist length

    Returns:
        Normalized intent with consistent schema
    """
    logger.info(f"Parsing intent: '{intent_text}' (length: {length})")
    # Check for top_tracks keywords first
    if _is_top_tracks_intent(intent_text):
        logger.info("Detected top_tracks mode from keywords")
        return _create_top_tracks_normalized(intent_text)

    # Parse with the configured LLM
    try:
        from app.agent.llm import model_name

        logger.info(f"Using LLM: {model_name()}")
        raw_intent = _call_ollama(intent_text, length)
        normalized = _normalize_raw_intent(raw_intent)
        logger.info("LLM intent parsing and normalization successful")
    except Exception as e:
        logger.warning(f"LLM parsing failed: {e}, using fallback")
        normalized = _create_fallback_normalized(intent_text)
    return _with_personal(_with_arc(_with_era(normalized, intent_text), intent_text), intent_text)


_PERSONAL_RE = None


def is_personal_request(text: str) -> bool:
    """Does the request ask for the listener's own taste ("my favorites", "what I like")?"""
    import re

    global _PERSONAL_RE
    if _PERSONAL_RE is None:
        _PERSONAL_RE = re.compile(
            r"\b(?:my (?:favou?rites?|taste|style|kind of|music|usual|top|library|playlists?)"
            r"|i (?:already )?(?:like|love|listen to|enjoy)|i'm into|what i like|for my taste"
            r"|like my|similar to (?:what|my))\b"
        )
    return bool(_PERSONAL_RE.search((text or "").lower()))


def _with_personal(intent: IntentNormalized, intent_text: str) -> IntentNormalized:
    intent.structure = {**(intent.structure or {}), "personal": is_personal_request(intent_text)}
    return intent


def _with_arc(intent: IntentNormalized, intent_text: str) -> IntentNormalized:
    """Replace the phase plan when the request describes an energy arc.

    The LLM schema only knows warmup/cruise/landing (a peak), so "builds into
    dancing" or "upbeat to sleepy" would otherwise be sequenced as a peak.
    Requests without arc words keep the parser's phases.
    """
    from app.agent.sequencer import ARC_PHASES, parse_arc

    shape = parse_arc(intent_text)
    if shape:
        intent.structure = {**(intent.structure or {}), "phases": [dict(p) for p in ARC_PHASES[shape]], "arc": shape}
    return intent


def _with_era(intent: IntentNormalized, intent_text: str) -> IntentNormalized:
    """Add a year_range constraint for decades named in the request ("80s", "nineties").

    Deterministic on purpose: the LLM schema has no era field and small models
    drop it from the description, yet it is one of the most explicit asks.
    """
    from app.catalog.base import parse_era

    era = parse_era(intent_text)
    if era:
        intent.constraints["year_range"] = list(era)
    return intent


def _is_top_tracks_intent(intent_text: str) -> bool:
    """Check if intent text indicates top tracks mode."""
    text_lower = intent_text.lower()
    keywords = ["top tracks", "most played", "most heard", "favorite", "best", "my top", "my favorite", "my most", "most listened"]

    return any(keyword in text_lower for keyword in keywords)


def _create_top_tracks_normalized(intent_text: str) -> IntentNormalized:
    """Create normalized intent for top tracks mode."""
    return IntentNormalized(
        mode="top_tracks",
        description="User's most played tracks",
        constraints={
            "energy_range": [0.0, 1.0],
            "valence_range": [0.0, 1.0],
            "bpm_range": [60.0, 180.0],
            "danceability_min": None
        },
        structure={"phases": []}  # No phases for top tracks
    )


def _call_ollama(intent_text: str, length: int) -> dict:
    """Call Ollama via LangChain LCEL chain to parse intent.

    Uses the configured chat model + ChatPromptTemplate + JsonOutputParser from llm.py,
    replacing direct HTTP calls with a composable LangChain chain.
    """
    from app.agent.llm import invoke_intent_chain

    try:
        parsed = invoke_intent_chain(intent_text, length)
        return parsed
    except Exception as e:
        raise Exception(f"LangChain intent chain failed: {e}")


def _normalize_raw_intent(raw_intent) -> IntentNormalized:
    """Normalize raw intent from Ollama into strict schema."""
    # Handle case where response might be a list or unexpected format
    if not isinstance(raw_intent, dict):
        logger.warning(f"Unexpected response type from Ollama: {type(raw_intent)}, using fallback")
        raise Exception("Unexpected response format")

    # Extract description
    description = raw_intent.get("description", "Parsed playlist")

    # Normalize constraints
    raw_constraints = raw_intent.get("constraints", {})
    constraints = _normalize_constraints(raw_constraints)

    # Normalize structure
    raw_structure = raw_intent.get("structure", {})
    structure = _normalize_structure(raw_structure)

    return IntentNormalized(
        mode="curate",
        description=description,
        constraints=constraints,
        structure=structure
    )


def _normalize_constraints(raw_constraints: dict) -> dict:
    """Normalize constraints into range tuples with no nulls."""
    # Energy range
    energy_min = max(0.0, min(1.0, raw_constraints.get("energy_min") or 0.0))
    energy_max = max(0.0, min(1.0, raw_constraints.get("energy_max") or 1.0))
    if energy_min > energy_max:
        energy_min, energy_max = energy_max, energy_min

    # Valence range
    valence_min = max(0.0, min(1.0, raw_constraints.get("valence_min") or 0.0))
    valence_max = max(0.0, min(1.0, raw_constraints.get("valence_max") or 1.0))
    if valence_min > valence_max:
        valence_min, valence_max = valence_max, valence_min

    # BPM range
    bpm_min = max(40.0, min(220.0, raw_constraints.get("tempo_min") or 60.0))
    bpm_max = max(40.0, min(220.0, raw_constraints.get("tempo_max") or 180.0))
    if bpm_min > bpm_max:
        bpm_min, bpm_max = bpm_max, bpm_min

    return {
        "energy_range": [energy_min, energy_max],
        "valence_range": [valence_min, valence_max],
        "bpm_range": [bpm_min, bpm_max],
        "danceability_min": raw_constraints.get("danceability_min")
    }


def _normalize_structure(raw_structure) -> dict:
    """Normalize structure into phases list with percentages."""
    # Handle case where structure might be a list directly
    if isinstance(raw_structure, list):
        phases_list = raw_structure
    elif isinstance(raw_structure, dict):
        phases_list = raw_structure.get("phases", [])
    else:
        phases_list = []

    if not phases_list:
        # Default phases if none provided
        phases = [
            {"name": "warmup", "pct": 0.2, "energy_target": 0.4},
            {"name": "cruise", "pct": 0.6, "energy_target": 0.7},
            {"name": "landing", "pct": 0.2, "energy_target": 0.3}
        ]
    else:
        # Convert phase names to phase objects
        phases = []
        num_phases = len(phases_list)

        # Default percentages and energy targets
        default_pcts = {"warmup": 0.2, "cruise": 0.6, "landing": 0.2}
        default_energy_targets = {"warmup": 0.4, "cruise": 0.7, "landing": 0.3}

        for phase_name in phases_list:
            if isinstance(phase_name, str):
                pct = default_pcts.get(phase_name, 1.0 / num_phases)
                energy_target = default_energy_targets.get(phase_name, 0.5)
                phases.append({
                    "name": phase_name,
                    "pct": pct,
                    "energy_target": energy_target
                })
            else:
                # If already a dict, use as-is but ensure required fields
                phase_dict = dict(phase_name) if isinstance(phase_name, dict) else {"name": str(phase_name)}
                phase_dict.setdefault("pct", 1.0 / num_phases)
                phase_dict.setdefault("energy_target", 0.5)
                phases.append(phase_dict)

        # Ensure percentages sum to 1.0
        total_pct = sum(phase["pct"] for phase in phases)
        if abs(total_pct - 1.0) > 0.001:  # Allow small floating point differences
            # Renormalize to ensure exact sum of 1.0
            for phase in phases:
                phase["pct"] = round(phase["pct"] / total_pct, 6)

    return {"phases": phases}


def _create_fallback_normalized(intent_text: str) -> IntentNormalized:
    """Create fallback normalized intent when LLM fails."""
    return IntentNormalized(
        mode="curate",
        description=f"Playlist based on: {intent_text}",
        constraints={
            "energy_range": [0.0, 1.0],
            "valence_range": [0.0, 1.0],
            "bpm_range": [60.0, 180.0],
            "danceability_min": None
        },
        structure={
            "phases": [
                {"name": "warmup", "pct": 0.2, "energy_target": 0.4},
                {"name": "cruise", "pct": 0.6, "energy_target": 0.7},
                {"name": "landing", "pct": 0.2, "energy_target": 0.3}
            ]
        }
    )
