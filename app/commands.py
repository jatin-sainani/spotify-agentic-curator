"""CLI commands for spotify-agentic-curator."""

import json
from pathlib import Path
from typing import List, Optional

import typer

from app.spotify.auth import SpotifyAuth
from app.spotify.client import SpotifyClient
from core.config import project_path, settings
from core.logging import logger

# Main command apps
connect_app = typer.Typer(help="Connect to Spotify API")
tools_app = typer.Typer(help="Test tools and integrations")
llm_app = typer.Typer(help="Test LLM integrations")
curate_app = typer.Typer(help="Curate playlists")
draft_app = typer.Typer(help="Manage playlist drafts")
catalog_app = typer.Typer(help="Build and inspect the offline track catalog")
eval_app = typer.Typer(help="Evaluate curation quality (gold prompts, metrics, LLM judge)")
mcp_app = typer.Typer(help="Expose the curator to other AI agents over MCP")


# Connect commands
# @app.command()
# def connect():
#     """Connect your Spotify account (OAuth)."""
#     typer.echo("Not implemented: connect")

@connect_app.command("auth")
def connect_auth():
    """Authenticate with Spotify using OAuth Authorization Code flow with PKCE."""
    logger.info("Connect auth command called")

    try:
        auth = SpotifyAuth()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        typer.echo(f"Configuration error: {e}")
        raise typer.Exit(1)

    try:
        if auth.authenticate():
            typer.echo("Successfully authenticated with Spotify!")
            typer.echo("You can now use other commands that require Spotify access.")
        else:
            typer.echo("Authentication failed")
            raise typer.Exit(1)
    except ValueError as e:
        logger.error(f"Authentication error: {e}")
        typer.echo(f"Authentication failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}")
        typer.echo(f"Unexpected error: {e}")
        raise typer.Exit(1)


@connect_app.command("status")
def connect_status():
    """Check whether Spotify tokens exist and still work."""
    import time

    try:
        auth = SpotifyAuth()
    except ValueError as e:
        typer.echo(f"Not configured: {e}")
        raise typer.Exit(1)

    tokens = auth._load_tokens()
    if not tokens:
        typer.echo("Not connected. Run: python -m app connect auth")
        raise typer.Exit(1)

    remaining = int(tokens.get("expires_at", 0) - time.time())
    status = f"valid for {remaining // 60} min" if remaining > 0 else "expired, will refresh"
    typer.echo(f"Token cache: found ({status})")
    try:
        with SpotifyClient(auth) as client:
            me = client.get_me()
        typer.echo(f"Connected as: {me.get('display_name') or me.get('id')} ({me.get('product', 'unknown')} plan)")
    except Exception as e:
        typer.echo(f"Token present but the API call failed: {e}")
        raise typer.Exit(1)


# LLM test commands
@llm_app.command("test")
def llm_test(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to test intent parsing"),
    length: int = typer.Option(20, "--length", "-l", help="Playlist length for context"),
):
    """Test LLM intent parsing with a prompt."""
    from app.agent.intent import parse_intent

    logger.info(f"LLM test command called with prompt: '{prompt}', length: {length}")

    try:
        normalized_intent = parse_intent(prompt, length)

        typer.echo("Intent parsing successful!")
        from app.agent.llm import model_name
        typer.echo(f"Model: {model_name()}")
        typer.echo("\n=== NORMALIZED INTENT ===")
        typer.echo(normalized_intent.model_dump_json(indent=2))

    except Exception as e:
        logger.error(f"Intent parsing failed: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


def _create_mock_candidates() -> List["TrackRef"]:  # noqa: F821 (imported inside to avoid cycles)
    """Create mock candidates for testing."""
    # Import here to avoid circular imports
    from app.core.draft_models import TrackRef

    candidates_data = [
        ("Bohemian Rhapsody", ["Queen"], {"energy": 0.4, "tempo": 72, "valence": 0.3, "danceability": 0.4}),
        ("Blinding Lights", ["The Weeknd"], {"energy": 0.8, "tempo": 171, "valence": 0.3, "danceability": 0.5}),
        ("Stairway to Heaven", ["Led Zeppelin"], {"energy": 0.3, "tempo": 82, "valence": 0.4, "danceability": 0.3}),
        ("Uptown Funk", ["Mark Ronson", "Bruno Mars"], {"energy": 0.9, "tempo": 115, "valence": 0.9, "danceability": 0.8}),
        ("Billie Jean", ["Michael Jackson"], {"energy": 0.8, "tempo": 117, "valence": 0.8, "danceability": 0.9}),
        ("Hotel California", ["Eagles"], {"energy": 0.5, "tempo": 75, "valence": 0.4, "danceability": 0.5}),
        ("Shape of You", ["Ed Sheeran"], {"energy": 0.7, "tempo": 96, "valence": 0.8, "danceability": 0.8}),
        ("Smells Like Teen Spirit", ["Nirvana"], {"energy": 0.9, "tempo": 117, "valence": 0.5, "danceability": 0.5}),
        ("Wonderwall", ["Oasis"], {"energy": 0.6, "tempo": 87, "valence": 0.4, "danceability": 0.4}),
        ("Hey Jude", ["The Beatles"], {"energy": 0.4, "tempo": 147, "valence": 0.6, "danceability": 0.4}),
    ]

    candidates = []
    for i, (name, artists, features) in enumerate(candidates_data):
        track_ref = TrackRef(
            uri=f"spotify:track:mock_{i}",
            name=name,
            artists=artists,
            id=f"mock_{i}",
            audio_features=features
        )
        candidates.append(track_ref)

    return candidates


@llm_app.command("rank")
def llm_rank(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Playlist prompt to test ranking"),
    length: int = typer.Option(5, "--length", "-l", help="Number of tracks to select"),
):
    """Test ranking pipeline with mock candidates."""
    from app.agent.intent import parse_intent
    from app.agent.ranker import rank_and_select

    logger.info(f"Ranker test command called with prompt: '{prompt}', length: {length}")

    try:
        # Parse intent
        intent = parse_intent(prompt, length)

        # Create mock candidates with audio features
        mock_candidates = _create_mock_candidates()

        # Mock top artists (normally from Spotify)
        top_artists = {"Queen", "The Weeknd", "Daft Punk", "Radiohead"}

        # Rank and select
        selected = rank_and_select(mock_candidates, intent, length, top_artists)

        typer.echo("Ranking test successful!")
        typer.echo(f"Selected {len(selected)}/{length} tracks")
        typer.echo("\n=== SELECTED TRACKS ===")

        for i, track in enumerate(selected, 1):
            typer.echo(f"{i}. {track.name} - {track.artists[0] if track.artists else 'Unknown'}")
            typer.echo(f"   {track.why}")
            if track.audio_features:
                energy = track.audio_features.get("energy", "N/A")
                tempo = track.audio_features.get("tempo", "N/A")
                valence = track.audio_features.get("valence", "N/A")
                typer.echo(f"   Features: energy={energy:.2f}, bpm={tempo}, valence={valence:.2f}")
            typer.echo()

    except Exception as e:
        logger.error(f"Ranking test failed: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


# Tools smoke test commands
@tools_app.command("spotify")
def tools_spotify():
    """Test Spotify API integration - print connected user and top 5 tracks."""
    logger.info("Tools spotify smoke test called")

    try:
        auth = SpotifyAuth()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        typer.echo(f"Configuration error: {e}")
        raise typer.Exit(1)

    if not auth.get_valid_access_token():
        typer.echo("Not authenticated with Spotify")
        typer.echo("Run 'python -m app connect auth' first")
        raise typer.Exit(1)

    try:
        with SpotifyClient(auth) as client:
            # Get user profile
            user_data = client.get_me()
            display_name = user_data.get("display_name", user_data.get("id", "Unknown User"))
            typer.echo(f"Connected as: {display_name}")

            # Get top tracks
            tracks = client.get_top_tracks(limit=5, time_range="long_term")
            #tracks = client.get_top_tracks(limit=5, time_range="short_term")
            if not tracks:
                typer.echo("No top tracks found")
            else:
                typer.echo("\nYour top 5 tracks (long term):")
                for i, track in enumerate(tracks, 1):
                    typer.echo(f"{i}. {track.name} - {track.artist}")

    except ValueError as e:
        logger.error(f"Spotify test failed: {e}")
        typer.echo(f"Test failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during Spotify test: {e}")
        typer.echo(f"Unexpected error: {e}")
        raise typer.Exit(1)


@tools_app.command("llm")
def tools_llm():
    """Check the configured LLM providers are reachable and can parse an intent."""
    from app.agent.llm import invoke_intent_chain, model_name, provider_available

    ok = True
    for role in ("agent", "judge"):
        up = provider_available(role)
        ok &= up
        typer.echo(f"{role:5s}: {model_name(role):40s} {'reachable' if up else 'UNREACHABLE'}")
    if not provider_available("agent"):
        raise typer.Exit(1)
    try:
        parsed = invoke_intent_chain("calm piano for studying", 10)
        typer.echo(f"Intent parse OK: {json.dumps(parsed)[:200]}")
    except Exception as e:
        typer.echo(f"Intent parse FAILED: {e}")
        raise typer.Exit(1)
    if not ok:
        raise typer.Exit(1)


@tools_app.command("build-priors")
def tools_build_priors(
    csv_path: str = typer.Option(..., "--csv", "-c", help="Path to songs CSV file"),
    output_path: str = typer.Option("app/agent/mood_priors.json", "--out", "-o", help="Output JSON path")
):
    """Build offline mood priors from Kaggle songs dataset."""
    from app.agent.mood_priors import build_priors_from_csv

    logger.info(f"Building mood priors from CSV: {csv_path}")

    try:
        priors_data = build_priors_from_csv(csv_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(priors_data, f, indent=2, ensure_ascii=False)

        typer.echo(f"SUCCESS: Mood priors built and saved to {output_path}")
        typer.echo(f"Moods processed: {len(priors_data['moods'])}")

    except Exception as e:
        logger.error(f"Failed to build mood priors: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)


@tools_app.command("priors-smoke")
def tools_priors_smoke():
    """Load mood_priors.json and print summary for 2 moods (sad, chill)."""
    from app.agent.mood_priors import load_priors

    logger.info("Loading mood priors for smoke test")

    try:
        priors = load_priors()

        if not priors:
            typer.echo("ERROR: No mood priors found")
            raise typer.Exit(1)

        typer.echo("SUCCESS: Mood priors loaded successfully")
        typer.echo(f"Available moods: {', '.join(priors.keys())}")
        typer.echo("")

        # Show summary for sad and chill moods
        for mood in ["sad", "chill"]:
            if mood in priors:
                mood_data = priors[mood]
                typer.echo(f"Mood: {mood.upper()}")

                top_genres = mood_data.get("top_genres", [])
                typer.echo(f"  Top genres ({len(top_genres)}): {', '.join(top_genres[:5])}...")

                top_keywords = mood_data.get("top_lyric_keywords", [])
                typer.echo(f"  Top lyric keywords ({len(top_keywords)}): {', '.join(top_keywords[:8])}...")

                ranges = mood_data.get("feature_ranges", {})
                if ranges:
                    range_strs = [f"{k}: {v['p10']:.2f}-{v['p90']:.2f}" for k, v in ranges.items()]
                    typer.echo(f"  Feature ranges: {', '.join(range_strs)}")

                typer.echo("")
            else:
                typer.echo(f"ERROR: Mood '{mood}' not found in priors")
                typer.echo("")

    except Exception as e:
        logger.error(f"Priors smoke test failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)


@tools_app.command("score-smoke")
def tools_score_smoke(
    intent: str = typer.Option("sad breakup songs", "--intent", "-i", help="Test intent")
):
    """Score 5 mock tracks for given intent to validate JSON parsing."""
    from app.agent.intent import parse_intent
    from app.agent.ranker import _stage1_pre_score, _stage2_ollama_score
    from app.core.draft_models import TrackRef

    logger.info(f"Score smoke test with intent: '{intent}'")

    try:
        # Parse intent
        normalized_intent = parse_intent(intent, 5)

        # Create 5 mock tracks
        mock_tracks = [
            TrackRef(
                uri="spotify:track:mock_1",
                name="Someone Like You",
                artists=["Adele"],
                id="mock_1",
                audio_features={"energy": 0.3, "tempo": 135, "valence": 0.2}
            ),
            TrackRef(
                uri="spotify:track:mock_2",
                name="All I Want",
                artists=["Kodaline"],
                id="mock_2",
                audio_features={"energy": 0.4, "tempo": 86, "valence": 0.3}
            ),
            TrackRef(
                uri="spotify:track:mock_3",
                name="Hurt",
                artists=["Nine Inch Nails"],
                id="mock_3",
                audio_features={"energy": 0.2, "tempo": 80, "valence": 0.1}
            ),
            TrackRef(
                uri="spotify:track:mock_4",
                name="The Night We Met",
                artists=["Lord Huron"],
                id="mock_4",
                audio_features={"energy": 0.3, "tempo": 87, "valence": 0.4}
            ),
            TrackRef(
                uri="spotify:track:mock_5",
                name="Skinny Love",
                artists=["Bon Iver"],
                id="mock_5",
                audio_features={"energy": 0.2, "tempo": 77, "valence": 0.2}
            )
        ]

        # Mock top artists for Stage 1
        top_artists = {"Adele", "Bon Iver"}

        typer.echo("Testing Stage 1 (pre-scoring)...")
        stage1_results = _stage1_pre_score(mock_tracks, normalized_intent, top_artists)

        typer.echo("Stage 1 results:")
        for track, score in stage1_results:
            typer.echo(f"  {track.name} - {track.artists[0]}: {score:.3f}")

        typer.echo("")
        typer.echo("Testing Stage 2 (Ollama scoring)...")

        # Take top 5 from Stage 1 for Stage 2
        stage2_input = stage1_results[:5]
        stage2_results = _stage2_ollama_score(stage2_input, normalized_intent)

        typer.echo("Stage 2 results:")
        for track, final_score, semantic_data in stage2_results:
            relevance = semantic_data.get("relevance", 0)
            mood_tags = semantic_data.get("mood_tags", [])
            why = semantic_data.get("why", "")
            typer.echo(f"  {track.name} - {track.artists[0]}:")
            typer.echo(f"    URI: {track.uri}")
            typer.echo(f"    Relevance: {relevance}/100, Mood tags: {mood_tags}")
            typer.echo(f"    Why: '{why}'")
            typer.echo(f"    Final score: {final_score:.3f}")

        typer.echo("")
        typer.echo("SUCCESS: Score smoke test completed successfully!")

    except Exception as e:
        logger.error(f"Score smoke test failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)


# Curate commands

CATALOG_OPTION_HELP = "Track source: 'offline' (local dataset index) or 'spotify' (live API)"


def _validate_length(length: Optional[int]) -> int:
    length = 30 if length is None else length
    if length < 10 or length > 100:
        typer.echo("Error: Length must be between 10 and 100 tracks")
        raise typer.Exit(1)
    return length


def _print_draft(draft, header: str) -> None:
    typer.echo(header)
    typer.echo(f"Draft ID: {draft.id}")
    typer.echo(f"Mode: {draft.mode}")
    typer.echo(f"Tracks: {len(draft.tracks)}/{draft.length}")
    typer.echo(f"Intent: {draft.intent_text}")
    typer.echo("")
    typer.echo("Selected tracks:")
    for i, track in enumerate(draft.tracks, 1):
        artists_str = ", ".join(track.artists) if track.artists else "Unknown Artist"
        typer.echo(f"{i:2d}. {track.name} - {artists_str}")
        if track.why:
            why_short = track.why[:80] + "..." if len(track.why) > 80 else track.why
            typer.echo(f"    {why_short}")
    typer.echo("")
    typer.echo(f"Use 'python -m app draft show --draft-id {draft.id}' to view full details")


@curate_app.command("new")
def curate_new(
    intent: str = typer.Option(..., "--intent", "-i", help="Natural language playlist description"),
    length: Optional[int] = typer.Option(None, "--length", "-l", help="Target playlist length (default: 30)"),
    catalog: Optional[str] = typer.Option(None, "--catalog", "-c", help=CATALOG_OPTION_HELP),
    profile: Optional[str] = typer.Option(None, "--profile", help="Offline listener profile (profiles/<name>.json)"),
):
    """Create a playlist draft with direct (non-graph) pipeline calls."""
    from app.agent.candidates import build_candidates
    from app.agent.draft_store import new_draft_id, save_draft
    from app.agent.intent import parse_intent
    from app.agent.ranker import rank_and_select
    from app.catalog import get_catalog
    from app.core.draft_models import Draft

    length = _validate_length(length)
    logger.info(f"Creating new playlist draft: '{intent}' (length: {length})")

    try:
        source = get_catalog(catalog, profile)
        normalized_intent = parse_intent(intent, length)
        logger.info(f"Intent parsed - mode: {normalized_intent.mode}")

        if normalized_intent.mode == "top_tracks":
            draft_tracks = source.get_user_taste(limit=length)
            for track in draft_tracks:
                track.why = "From your top tracks"
        else:
            candidates = build_candidates(
                source, limit=min(200, length * 3), intent=normalized_intent, intent_text=intent
            )
            draft_tracks = rank_and_select(candidates, normalized_intent, length, source.top_artists(),
                                           request_text=intent)

        draft = Draft(
            id=new_draft_id(),
            intent_text=intent,
            mode=normalized_intent.mode,
            length=length,
            tracks=draft_tracks,
            status="created",
        )
        save_draft(draft)
        logger.info(f"Draft {draft.id} created and saved with {len(draft_tracks)} tracks")
        _print_draft(draft, "Draft created successfully!")

    except Exception as e:
        logger.error(f"Draft creation failed: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


@curate_app.command("graph")
def curate_graph(
    intent: str = typer.Option(..., "--intent", "-i", help="Natural language playlist description"),
    length: Optional[int] = typer.Option(None, "--length", "-l", help="Target playlist length (default: 30)"),
    catalog: Optional[str] = typer.Option(None, "--catalog", "-c", help=CATALOG_OPTION_HELP),
    profile: Optional[str] = typer.Option(None, "--profile", help="Offline listener profile (profiles/<name>.json)"),
    retrieval: str = typer.Option("auto", "--retrieval", "-r",
                                  help="'agent' (tool-calling LLM), 'heuristic' (keyword search), or 'auto'"),
    sequence: bool = typer.Option(False, "--sequence", help="Order tracks along the request's energy arc"),
):
    """Create a playlist draft using the LangGraph StateGraph pipeline.

    parse_intent -> retrieve (tool-calling agent) -> stage1_score -> [stage2_score] -> select_tracks

    Stage 2 (LLM semantic scoring) is skipped via a conditional edge if the
    configured LLM provider is unavailable (graceful degradation).
    """
    from app.agent.draft_store import new_draft_id, save_draft
    from app.agent.graph import run_curation
    from app.catalog import get_catalog
    from app.core.draft_models import Draft

    if retrieval not in ("auto", "agent", "heuristic"):
        typer.echo("Error: --retrieval must be one of auto, agent, heuristic")
        raise typer.Exit(1)
    length = _validate_length(length)
    logger.info(f"[graph] Creating playlist via LangGraph: '{intent}' (length: {length})")

    try:
        source = get_catalog(catalog, profile)
        result = run_curation(intent_text=intent, length=length, catalog=source, retrieval=retrieval,
                              sequence=sequence)
        selected_tracks = result.get("selected_tracks", [])

        draft = Draft(
            id=new_draft_id(),
            intent_text=intent,
            mode=result["intent"].mode if result.get("intent") else "curate",
            length=length,
            tracks=selected_tracks,
            status="created",
        )
        save_draft(draft)
        log = result.get("retrieval_log") or []
        _print_draft(draft, f"Draft created successfully! (via LangGraph, retrieval={result.get('retrieval_used')}, "
                            f"{len(log)} tool calls)")

    except Exception as e:
        logger.error(f"[graph] LangGraph curation failed: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


@curate_app.command("improve")
def curate_improve(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft to improve"),
    feedback: str = typer.Option(..., "--feedback", "-f", help="What to change, e.g. 'more upbeat, fewer covers'"),
    max_swaps: int = typer.Option(3, "--max-swaps", help="Maximum tracks to swap"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept every proposed swap without asking"),
    catalog: Optional[str] = typer.Option(None, "--catalog", "-c", help=CATALOG_OPTION_HELP),
    profile: Optional[str] = typer.Option(None, "--profile", help="Offline listener profile"),
):
    """Propose track swaps from feedback; you approve each one (human in the loop)."""
    from app.agent.draft_store import save_draft
    from app.agent.improve import improve_draft
    from app.catalog import get_catalog

    draft = _load_or_exit(draft_id)

    def decide(proposals):
        typer.echo(f"\n{len(proposals)} proposed swap(s) for: {feedback!r}\n")
        decisions = []
        for i, prop in enumerate(proposals, 1):
            old, new = prop["remove"], prop["add"]
            typer.echo(f"{i}. OUT {old['name']} - {', '.join(old['artists'])}  (score {prop['old_score']})")
            typer.echo(f"   IN  {new['name']} - {', '.join(new['artists'])}  (score {prop['new_score']})")
            decisions.append(True if yes else typer.confirm("   Accept this swap?", default=True))
        return decisions

    try:
        result = improve_draft(draft, feedback, get_catalog(catalog, profile), decide, max_swaps=max_swaps)
    except Exception as e:
        logger.error(f"Improve failed: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)

    if not result["proposals"]:
        typer.echo("No better-fitting replacements found for that feedback; draft unchanged.")
        return
    save_draft(result["draft"])
    typer.echo(f"\nApplied {result['applied']}/{len(result['proposals'])} swaps to draft {draft_id}.")


# Draft commands with subcommands
@draft_app.command("show")
def draft_show(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft ID to show"),
):
    """Show draft playlist details."""
    from app.agent.draft_store import list_drafts, load_draft

    logger.info(f"Draft show command called for draft: {draft_id}")

    try:
        draft = load_draft(draft_id)

        # Display draft header info
        typer.echo(f"Draft ID: {draft.id}")
        typer.echo(f"Status: {getattr(draft.status, 'value', draft.status)}")
        typer.echo(f"Mode: {draft.mode}")
        typer.echo(f"Created: {draft.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        typer.echo(f"Intent: {draft.intent_text}")
        typer.echo(f"Target Length: {draft.length}")
        typer.echo(f"Current Tracks: {len(draft.tracks)}")
        if draft.external_url:
            typer.echo(f"Spotify: {draft.external_url}")
        typer.echo("")

        if not draft.tracks:
            typer.echo("No tracks in this draft yet.")
            return

        # Display tracks
        typer.echo("Tracks:")
        for i, track in enumerate(draft.tracks, 1):
            artists_str = ", ".join(track.artists)
            typer.echo(f"{i:2d}. {track.name} - {artists_str}  [{track.id}]")
            if track.why:
                typer.echo(f"    Why: {track.why}")

        # Display feedback events if any
        if draft.feedback_events:
            typer.echo("")
            typer.echo(f"Feedback Events: {len(draft.feedback_events)}")

    except FileNotFoundError:
        logger.error(f"Draft {draft_id} not found")
        typer.echo(f"Error: Draft '{draft_id}' not found")
        available_drafts = list_drafts()
        if available_drafts:
            typer.echo(f"Available drafts: {', '.join(available_drafts[:5])}")
            if len(available_drafts) > 5:
                typer.echo(f"... and {len(available_drafts) - 5} more")
        else:
            typer.echo("No drafts available")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to show draft {draft_id}: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


@draft_app.command("list")
def draft_list():
    """List all available drafts."""
    from app.agent.draft_store import list_drafts

    logger.info("Draft list command called")

    try:
        drafts = list_drafts()

        if not drafts:
            typer.echo("No drafts found.")
            return

        typer.echo(f"Available drafts ({len(drafts)}):")
        for draft_id in drafts:
            typer.echo(f"  {draft_id}")

    except Exception as e:
        logger.error(f"Failed to list drafts: {e}")
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


def _load_or_exit(draft_id: str):
    from app.agent.draft_store import load_draft

    try:
        return load_draft(draft_id)
    except FileNotFoundError:
        typer.echo(f"Error: Draft '{draft_id}' not found (see: python -m app draft list)")
        raise typer.Exit(1)


def _apply_edit(draft_id: str, edit, success: str) -> None:
    """Load a draft, apply an edit function, save, and report."""
    from app.agent.draft_ops import DraftError
    from app.agent.draft_store import save_draft

    draft = _load_or_exit(draft_id)
    try:
        edit(draft)
    except DraftError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)
    save_draft(draft)
    typer.echo(success)


@draft_app.command("remove")
def draft_remove(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft ID"),
    track_id: str = typer.Option(..., "--track-id", "-t", help="Track ID to remove"),
):
    """Remove a track from a draft."""
    from app.agent.draft_ops import remove_track

    _apply_edit(draft_id, lambda d: remove_track(d, track_id), f"Removed {track_id} from draft {draft_id}")


@draft_app.command("move")
def draft_move(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft ID"),
    track_id: str = typer.Option(..., "--track-id", "-t", help="Track ID to move"),
    position: int = typer.Option(..., "--position", "-p", help="New 1-based position"),
):
    """Move a track to a new position in a draft."""
    from app.agent.draft_ops import move_track

    _apply_edit(draft_id, lambda d: move_track(d, track_id, position), f"Moved {track_id} to position {position}")


@draft_app.command("add")
def draft_add(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft ID"),
    track_id: str = typer.Option(..., "--track-id", "-t", help="Track ID to add"),
    position: Optional[int] = typer.Option(None, "--position", "-p", help="1-based position (default: end)"),
    catalog: Optional[str] = typer.Option(None, "--catalog", "-c", help=CATALOG_OPTION_HELP),
):
    """Add a track (looked up in the catalog) to a draft."""
    from app.agent.draft_ops import add_track
    from app.catalog import get_catalog

    found = get_catalog(catalog).get_tracks([track_id])
    if not found:
        typer.echo(f"Error: Track '{track_id}' not found in the catalog")
        raise typer.Exit(1)
    _apply_edit(draft_id, lambda d: add_track(d, found[0], position),
                f"Added {found[0].name} - {', '.join(found[0].artists)} to draft {draft_id}")


@draft_app.command("rerank")
def draft_rerank(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft ID to rerank"),
    criteria: str = typer.Option(..., "--criteria", "-c", help="How to reorder, e.g. 'build energy toward the end'"),
):
    """Reorder a draft's tracks for new criteria (membership unchanged)."""
    from app.agent.draft_ops import rerank

    _apply_edit(draft_id, lambda d: rerank(d, criteria), f"Reranked draft {draft_id} for: {criteria}")


@draft_app.command("confirm")
def draft_confirm(
    draft_id: str = typer.Option(..., "--draft-id", "-d", help="Draft ID to confirm"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Playlist name (default: the request text)"),
    description: Optional[str] = typer.Option(None, "--description", help="Playlist description"),
    target: Optional[str] = typer.Option(None, "--target",
                                         help="'spotify' (create the playlist) or 'file' (export). Default follows CATALOG"),
    fmt: str = typer.Option("uris", "--format", help="File export format: 'uris' (paste into Spotify) or 'json'"),
    public: bool = typer.Option(False, "--public", help="Make the Spotify playlist public"),
):
    """Finalize a draft: create the Spotify playlist, or export it to a file."""
    from app.agent.draft_ops import DraftError, confirm_to_spotify, export_draft
    from app.agent.draft_store import save_draft

    draft = _load_or_exit(draft_id)
    target = (target or ("spotify" if settings.catalog == "spotify" else "file")).lower()

    try:
        if target == "file":
            path = export_draft(draft, fmt=fmt)
            typer.echo(f"Exported {len(draft.tracks)} tracks to {path}")
            if fmt == "uris":
                typer.echo("Tip: select-all and paste these URIs into a playlist in the Spotify desktop app.")
            return
        if target != "spotify":
            typer.echo("Error: --target must be 'spotify' or 'file'")
            raise typer.Exit(1)

        playlist_name = name or draft.intent_text[:100]
        with SpotifyClient(SpotifyAuth()) as client:
            confirm_to_spotify(draft, client, playlist_name,
                                          description or f"Curated by Spotify Agentic Curator: {draft.intent_text}"[:300],
                                          public=public)
        save_draft(draft)
        typer.echo(f"Created playlist '{playlist_name}' with {len(draft.tracks)} tracks")
        if draft.external_url:
            typer.echo(draft.external_url)
    except (DraftError, ValueError) as e:
        typer.echo(f"Error: {e}")
        if target == "spotify":
            typer.echo("Spotify Dev Mode apps need a Premium account and an allow-listed user. "
                       "Fallback: --target file")
        raise typer.Exit(1)


# Catalog commands
@catalog_app.command("build")
def catalog_build(
    csv_path: str = typer.Option(..., "--csv", help="Raw songs dataset CSV"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output index path (.parquet or .csv)"),
    sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Keep a genre-stratified sample of N tracks"),
):
    """Build the offline catalog index from a songs dataset."""
    from app.catalog.offline import build_index

    out_path = out or settings.catalog_path
    try:
        df = build_index(csv_path, out_path, sample=sample)
    except ImportError as e:
        typer.echo(f"Error: {e}\nInstall pyarrow for .parquet output, or pass --out data/catalog.csv")
        raise typer.Exit(1)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)
    typer.echo(f"Built offline catalog: {len(df)} tracks -> {out_path}")
    typer.echo(f"Genres: {df['genre'].value_counts().head(10).to_dict()}")


@catalog_app.command("search")
def catalog_search(
    query: str = typer.Argument(..., help="Search text, e.g. 'acoustic folk rainy'"),
    limit: int = typer.Option(10, "--limit", "-l"),
    catalog: Optional[str] = typer.Option(None, "--catalog", "-c", help=CATALOG_OPTION_HELP),
):
    """Search the catalog (useful for checking the index)."""
    from app.catalog import get_catalog

    source = get_catalog(catalog)
    for i, ref in enumerate(source.search(query, limit=limit), 1):
        f = ref.audio_features or {}
        typer.echo(
            f"{i:2d}. {ref.name} - {', '.join(ref.artists)} "
            f"[{', '.join(ref.genres[:2])}] energy={f.get('energy', '?')} valence={f.get('valence', '?')}"
        )


# Eval commands
def _results_path(run: str) -> Path:
    from evals.runner import RESULTS_DIR

    path = Path(run)
    return path if path.suffix == ".json" and path.exists() else RESULTS_DIR / f"{Path(run).stem}.json"


@eval_app.command("run")
def eval_run(
    name: str = typer.Option(..., "--name", "-n", help="Run name (results/<name>.json; reruns resume)"),
    configs: str = typer.Option("baseline,llm_rank,agent", "--configs", "-c", help="Comma-separated config names"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Comma-separated prompt categories"),
    prompt_ids: Optional[str] = typer.Option(None, "--prompts", help="Comma-separated prompt ids"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max prompts (stratified across categories)"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Grade playlists with the LLM judge"),
    passes: int = typer.Option(2, "--passes", help="Judge passes (2nd+ use shuffled track order)"),
):
    """Run configs over the gold prompts (offline catalog) and write a report."""
    from app.agent.llm import model_name
    from app.catalog.offline import OfflineCatalog
    from evals.judge import PlaylistJudge
    from evals.report import build_report
    from evals.runner import CONFIGS, EvalRunner, load_prompts, select_prompts

    unknown = [c for c in configs.split(",") if c not in CONFIGS]
    if unknown:
        typer.echo(f"Error: unknown configs {unknown}. Available: {', '.join(CONFIGS)}")
        raise typer.Exit(1)
    prompts = select_prompts(
        load_prompts(),
        ids=prompt_ids.split(",") if prompt_ids else None,
        categories=categories.split(",") if categories else None,
        limit=limit,
    )
    if not prompts:
        typer.echo("Error: no prompts matched")
        raise typer.Exit(1)

    runner = EvalRunner(
        catalog_factory=lambda profile: OfflineCatalog(str(project_path(settings.catalog_path)), profile=profile),
        judge=PlaylistJudge(passes=passes) if judge else None,
    )
    meta = {"agent_model": model_name("agent"), "judge_model": model_name("judge") if judge else None,
            "catalog": settings.catalog_path}
    typer.echo(f"Running {len(prompts)} prompts x {len(configs.split(','))} configs -> evals/results/{name}.json")
    results = runner.run(prompts, [CONFIGS[c] for c in configs.split(",")], name, meta=meta)

    report_path = _results_path(name).with_suffix(".md")
    report_path.write_text(build_report(results), encoding="utf-8")
    typer.echo(f"Report: {report_path}")


@eval_app.command("report")
def eval_report(
    run: str = typer.Argument(..., help="Run name or path to results JSON"),
    baseline: str = typer.Option("baseline", "--baseline", help="Config to compare against"),
):
    """Rebuild the Markdown report for a results file."""
    from evals.report import build_report

    path = _results_path(run)
    results = json.loads(path.read_text(encoding="utf-8"))
    report = build_report(results, baseline=baseline)
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    typer.echo(report)


@eval_app.command("label")
def eval_label(
    run: str = typer.Argument(..., help="Run name or path to results JSON"),
    n: int = typer.Option(40, "--n", help="Tracks to sample for labeling"),
):
    """Write a blind CSV of sampled tracks for human relevance labels (0 / 0.5 / 1)."""
    from evals.agreement import write_label_sheet
    from evals.runner import EVALS_DIR

    results = json.loads(_results_path(run).read_text(encoding="utf-8"))
    out = EVALS_DIR / "human_labels" / f"{results['run']}.csv"
    if out.exists():
        typer.echo(f"Error: {out} exists; delete it first to resample (it may hold your labels)")
        raise typer.Exit(1)
    count = write_label_sheet(results, out, n=n)
    typer.echo(f"Wrote {count} tracks to {out}")
    typer.echo("Fill the 'label' column with 0, 0.5 or 1, then run: python -m app eval agreement " + results["run"])


@eval_app.command("agreement")
def eval_agreement(
    run: str = typer.Argument(..., help="Run name or path to results JSON"),
    labels: Optional[str] = typer.Option(None, "--labels", help="Labels CSV (default: evals/human_labels/<run>.csv)"),
):
    """Compare judge labels with human labels (weighted kappa, Spearman)."""
    from evals.agreement import agreement_report, load_labels, pair_labels
    from evals.runner import EVALS_DIR

    results = json.loads(_results_path(run).read_text(encoding="utf-8"))
    path = Path(labels) if labels else EVALS_DIR / "human_labels" / f"{results['run']}.csv"
    report = agreement_report(pair_labels(results, load_labels(path)))
    if not report["n"]:
        typer.echo(f"No labeled rows found in {path}")
        raise typer.Exit(1)
    for k, v in report.items():
        typer.echo(f"{k:16s} {v:.3f}" if isinstance(v, float) else f"{k:16s} {v}")


# MCP server
@mcp_app.command("serve")
def mcp_serve():
    """Serve the curator over MCP (stdio). Point an MCP client at: python -m app mcp serve"""
    from app.mcp_server import serve

    serve()


@eval_app.command("injection")
def eval_injection(
    name: str = typer.Option("injection", "--name", "-n", help="Run name (results/<name>.json; reruns resume)"),
    attacks: Optional[str] = typer.Option(None, "--attacks", help="Comma-separated: score_inflation, "
                                          "agent_hijack, id_injection, judge_inflation"),
    modes: str = typer.Option("off,prompt_only,all_on", "--modes", help="Defense modes to compare"),
    fpr_csv: Optional[str] = typer.Option(None, "--fpr-csv", help="Also measure false positives on this catalog file"),
):
    """Red-team the pipeline with prompt injections planted in catalog metadata."""
    from app.agent.llm import model_name
    from evals.injection.run import RESULTS_DIR, build_report, false_positive_rate, run_suite
    from evals.injection.scenarios import DEFENSE_MODES

    bad = [m for m in modes.split(",") if m not in DEFENSE_MODES]
    if bad:
        typer.echo(f"Error: unknown modes {bad}. Available: {', '.join(DEFENSE_MODES)}")
        raise typer.Exit(1)
    results = run_suite(name, attacks=attacks.split(",") if attacks else None, modes=modes.split(","),
                        meta={"model": model_name("agent"), "judge_model": model_name("judge")})
    if fpr_csv:
        results["false_positives"] = false_positive_rate(fpr_csv)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
    report = build_report(results)
    (RESULTS_DIR / f"{name}.md").write_text(report, encoding="utf-8")
    typer.echo(report)
