"""LangChain LLM integration for playlist curation.

Provides LCEL (LangChain Expression Language) chains for:
- Intent parsing: natural language → structured IntentNormalized schema
- Track scoring: track + intent → relevance score with mood tags
"""

from typing import Any, Dict, Literal

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from core.logging import logger

Role = Literal["agent", "judge"]
SUPPORTED_PROVIDERS = ("ollama", "anthropic")


def resolve_provider(role: Role = "agent") -> str:
    """Return the provider configured for a role ("agent" or "judge")."""
    provider = settings.llm_provider
    if role == "judge" and settings.judge_provider:
        provider = settings.judge_provider
    provider = (provider or "ollama").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        logger.warning(
            f"Unsupported LLM provider '{provider}', falling back to 'ollama' "
            f"(supported: {', '.join(SUPPORTED_PROVIDERS)})"
        )
        provider = "ollama"
    return provider


def get_llm(
    temperature: float = 0.7,
    role: Role = "agent",
    json_mode: bool = True,
) -> BaseChatModel:
    """Create a chat model for the given role.

    Args:
        temperature: Sampling temperature.
        role: "agent" for curation, "judge" for evaluation.
        json_mode: Ask the model for JSON output. Ollama supports a native JSON
            format flag; for Anthropic the prompts already demand JSON and
            JsonOutputParser extracts it. Disable for tool-calling.
    """
    provider = resolve_provider(role)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("LLM provider is 'anthropic' but ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=2048,
        )

    from langchain_ollama import ChatOllama

    kwargs: Dict[str, Any] = {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def provider_available(role: Role = "agent") -> bool:
    """Cheap reachability check for the provider configured for a role.

    Ollama: the local server answers /api/tags. Anthropic: an API key is set
    (network errors are handled by the callers' fallbacks).
    """
    provider = resolve_provider(role)
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def model_name(role: Role = "agent") -> str:
    """Human-readable "<provider>:<model>" for logs and eval reports."""
    provider = resolve_provider(role)
    model = settings.anthropic_model if provider == "anthropic" else settings.ollama_model
    return f"{provider}:{model}"


# ---------------------------------------------------------------------------
# Intent Parsing Chain
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are an expert music curator. Parse the user's playlist request into a JSON format.

Respond with ONLY valid JSON. No markdown, no explanations, just JSON.

Required JSON schema:
{{
  "description": "brief human-readable description",
  "constraints": {{
    "energy_min": number 0.0-1.0 or null,
    "energy_max": number 0.0-1.0 or null,
    "valence_min": number 0.0-1.0 or null,
    "valence_max": number 0.0-1.0 or null,
    "tempo_min": number > 0 or null,
    "tempo_max": number > 0 or null,
    "danceability_min": number 0.0-1.0 or null
  }},
  "structure": {{
    "phases": ["warmup", "cruise", "landing"] or []
  }}
}}"""

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", "Create a {length}-track playlist for: {intent_text}"),
])


def build_intent_chain():
    """Build the intent parsing LCEL chain.

    Chain: prompt → chat model → JsonOutputParser → dict
    """
    return INTENT_PROMPT | get_llm() | JsonOutputParser()


def invoke_intent_chain(intent_text: str, length: int) -> dict:
    """Parse a natural-language intent into structured JSON using LangChain.

    Args:
        intent_text: User's playlist description
        length: Target playlist length

    Returns:
        Parsed intent dict with description, constraints, and structure
    """
    chain = build_intent_chain()
    logger.info(f"Invoking LangChain intent chain (model: {model_name()})")
    return chain.invoke({"intent_text": intent_text, "length": str(length)})


# ---------------------------------------------------------------------------
# Batch Track Scoring Chain
# ---------------------------------------------------------------------------

SCORE_SYSTEM_PROMPT = """\
You are an expert music curator. Score how well each candidate track fits a playlist request.

Scoring guide (relevance 0-100):
- 90-100: an obvious, ideal pick for this exact request
- 60-89: fits the mood/genre well
- 30-59: partially fits (right genre but wrong mood, or vice versa)
- 0-29: clearly does not fit

Use the genre labels and audio features (energy, valence = musical positivity, tempo,
acousticness; all 0-1 except tempo in BPM) as evidence, not just the title.
{untrusted_notice}
Respond with ONLY valid JSON, no markdown:
{{"scores": [{{"n": <track number>, "relevance": <0-100>, "mood_tags": ["tag", "tag"], "why": "<max 12 words>"}}]}}
Include exactly one entry per track number."""

SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SCORE_SYSTEM_PROMPT),
    ("human",
     "Playlist request: {description}\n{priors_context}\n\n"
     "Candidate tracks:\n{tracks_block}"),
])


def build_score_chain():
    """Build the batch track scoring LCEL chain.

    Chain: prompt → chat model → JsonOutputParser → dict
    """
    # Temperature 0: scores feed ranking and evaluation, so consistency beats creativity.
    return SCORE_PROMPT | get_llm(temperature=0.0) | JsonOutputParser()


def invoke_score_chain(
    tracks_block: str,
    intent_description: str,
    priors_context: str = "",
) -> dict:
    """Score a numbered block of tracks in one LLM call.

    Args:
        tracks_block: One line per track, each starting with its number "n".
        intent_description: Human-readable playlist intent.
        priors_context: Optional mood priors context string.

    Returns:
        Dict shaped like {"scores": [{"n", "relevance", "mood_tags", "why"}, ...]}
    """
    from app.agent.safety import untrusted_notice, wrap_catalog_data

    chain = build_score_chain()
    notice = untrusted_notice()
    return chain.invoke({
        "description": intent_description,
        "priors_context": priors_context,
        "tracks_block": wrap_catalog_data(tracks_block),
        "untrusted_notice": f"\n{notice}\n" if notice else "",
    })
