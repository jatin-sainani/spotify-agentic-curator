"""Tests for the tool-calling retrieval agent and the curation graph (scripted fake LLM)."""

from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.agent.graph import run_curation
from app.agent.intent import IntentNormalized
from app.agent.retrieval import agent_retrieve, merge_pools
from app.agent.tools import RetrievalSession, create_retrieval_tools, describe_track
from app.catalog.offline import OfflineCatalog

FIXTURE = Path(__file__).parent / "fixtures" / "catalog_synthetic.csv"
PROFILE = {"name": "test", "seed_artists": ["Gravel Kings"]}


class ScriptedToolLLM(FakeMessagesListChatModel):
    """Fake chat model that replays scripted AIMessages and accepts bind_tools."""

    bound_tools: List[Any] = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [t.name for t in tools]
        return self


def run_tool(tool, call_id="c1", **args):
    """Invoke a tool the way ToolNode does (a full ToolCall) and return its text."""
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": call_id}).content


def tool_call(name, **args):
    return {"name": name, "args": args, "id": f"call_{name}_{len(args)}_{abs(hash(str(args))) % 9999}"}


@pytest.fixture(scope="module")
def catalog():
    return OfflineCatalog(str(FIXTURE), profile=PROFILE)


def intent(mode="curate"):
    return IntentNormalized(
        mode=mode, description="quiet folk",
        constraints={"energy_range": [0.0, 0.5], "valence_range": [0.0, 1.0], "bpm_range": [60, 180],
                     "danceability_min": None},
        structure={"phases": []},
    )


# -- tools -----------------------------------------------------------------

def test_search_tool_records_pool_and_log(catalog):
    session = RetrievalSession(catalog=catalog)
    search = {t.name: t for t in create_retrieval_tools(session)}["search_catalog"]

    out = run_tool(search, query="folk", energy_max=0.5, limit=5)

    assert len([ln for ln in out.splitlines() if ln.startswith("[")]) == 5
    assert out.startswith("<catalog_data>")  # spotlighting on by default
    assert all(ref.audio_features["energy"] <= 0.5 for ref in session.pool.values())
    assert session.calls[0]["tool"] == "search_catalog" and session.calls[0]["results"] == 5
    assert set(session.sources.values()) == {"search:folk"}


def test_search_tool_caps_limit_and_handles_no_results(catalog):
    session = RetrievalSession(catalog=catalog)
    search = {t.name: t for t in create_retrieval_tools(session)}["search_catalog"]
    assert "No tracks found" in run_tool(search, query="zzzzqqq")
    run_tool(search, "c2", query="song", limit=999)
    assert len(session.pool) == 25


def test_taste_and_features_tools(catalog):
    session = RetrievalSession(catalog=catalog)
    tools = {t.name: t for t in create_retrieval_tools(session)}
    taste = run_tool(tools["get_user_taste"], limit=3)
    assert "Gravel Kings" in taste and len(session.pool) == 3
    feats = run_tool(tools["get_audio_features"], "c2", track_ids=["synth0001", "nope"])
    assert "synth0001" in feats and "nope" not in feats


def test_finalize_restores_model_call_order(catalog):
    # Simulate ToolNode threads finishing in reverse order of the model's calls.
    session = RetrievalSession(catalog=catalog)
    search = {t.name: t for t in create_retrieval_tools(session)}["search_catalog"]
    run_tool(search, "second", query="jazz", limit=2)
    run_tool(search, "first", query="folk", limit=2)
    assert [c["query"] for c in session.calls] == ["jazz", "folk"]

    session.finalize(["first", "second"])

    assert [c["query"] for c in session.calls] == ["folk", "jazz"]
    assert [r.genres[0] for r in session.pool.values()] == ["Folk", "Folk", "Jazz", "Jazz"]


def test_describe_track_includes_id_and_features(catalog):
    line = describe_track(catalog.get_tracks(["synth0001"])[0])
    assert line.startswith("[synth0001] Rain Song 1 - Willow Creek (Folk")
    assert "energy=" in line and "tempo=" in line


# -- agent loop --------------------------------------------------------------

def test_agent_runs_tools_until_done(catalog):
    llm = ScriptedToolLLM(responses=[
        AIMessage(content="", tool_calls=[tool_call("get_user_taste", limit=3)]),
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="indie folk"),
                                          tool_call("search_catalog", query="jazz", energy_max=0.4)]),
        AIMessage(content="Gathered folk and mellow jazz."),
    ])
    session = agent_retrieve(catalog, "quiet folk", "quiet folk", target=30, llm=llm)

    assert llm.bound_tools == ["search_catalog", "get_user_taste", "get_audio_features"]
    assert [c["tool"] for c in session.calls] == ["get_user_taste", "search_catalog", "search_catalog"]
    artists = {r.artists[0] for r in session.pool.values()}
    assert {"Gravel Kings", "Willow Creek"} <= artists


def test_agent_respects_step_budget(catalog):
    looping = [AIMessage(content="", tool_calls=[tool_call("search_catalog", query=f"q{i}")]) for i in range(20)]
    session = agent_retrieve(catalog, "x", "x", target=30, llm=ScriptedToolLLM(responses=looping), max_rounds=3)
    assert len(session.calls) == 2  # tools run after rounds 1 and 2; round 3 hits the budget


def test_agent_is_nudged_once_when_pool_is_thin(catalog):
    llm = ScriptedToolLLM(responses=[
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="house", limit=3)]),
        AIMessage(content="Done."),                      # stops with 3/30 -> nudged
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="bebop", limit=3)]),
        AIMessage(content="Done again."),                # still thin, but nudge budget spent
    ])
    session = agent_retrieve(catalog, "anything", "anything", target=30, llm=llm)
    assert [c["query"] for c in session.calls] == ["house", "bebop"]
    assert len(session.pool) == 6


def test_agent_not_nudged_when_pool_is_full_and_varied(catalog):
    llm = ScriptedToolLLM(responses=[
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="song", limit=20),
                                          tool_call("search_catalog", query="jazz", limit=10)]),
        AIMessage(content="Done."),
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="SHOULD-NOT-RUN")]),
    ])
    session = agent_retrieve(catalog, "x", "x", target=30, llm=llm)
    assert [c["query"] for c in session.calls] == ["song", "jazz"]


def test_agent_nudged_when_single_search_angle(catalog):
    llm = ScriptedToolLLM(responses=[
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="song", limit=25)]),
        AIMessage(content="Done."),  # 25/30 tracks, but only one angle -> nudged
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="bebop", limit=5)]),
        AIMessage(content="Done."),
    ])
    session = agent_retrieve(catalog, "x", "x", target=30, llm=llm)
    assert [c["query"] for c in session.calls] == ["song", "bebop"]


def test_agent_tool_errors_are_reported_not_raised(catalog):
    llm = ScriptedToolLLM(responses=[
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="folk", limit="not-a-number")]),
        AIMessage(content="done"),
    ])
    session = agent_retrieve(catalog, "folk", "folk", target=10, llm=llm)
    assert session.pool == {}


def test_merge_pools_dedupes_and_caps(catalog):
    a = catalog.get_tracks(["synth0001", "synth0002"])
    b = catalog.get_tracks(["synth0002", "synth0003", "synth0004"])
    assert [r.id for r in merge_pools(a, b, 3)] == ["synth0001", "synth0002", "synth0003"]


# -- full graph --------------------------------------------------------------

@pytest.fixture
def offline_llm_down():
    """Intent parsing uses its fallback; Stage 2 is skipped (no real LLM in tests)."""
    with patch("app.agent.intent._call_ollama", side_effect=RuntimeError("no llm")), \
         patch("app.agent.llm.provider_available", return_value=False):
        yield


def test_graph_heuristic_path(catalog, offline_llm_down):
    result = run_curation("folk for a rainy day", 10, catalog, retrieval="heuristic")
    assert result["retrieval_used"] == "heuristic"
    assert len(result["selected_tracks"]) == 10
    assert all(t.why for t in result["selected_tracks"])


def test_graph_agent_path_uses_agent_pool(catalog, offline_llm_down):
    llm = ScriptedToolLLM(responses=[
        AIMessage(content="", tool_calls=[tool_call("search_catalog", query="cool jazz", limit=12),
                                          tool_call("search_catalog", query="indie folk", limit=12)]),
        AIMessage(content="done"),
    ])
    with patch("app.agent.graph._llm_up", return_value=False):
        result = run_curation("jazz and folk", 10, catalog, retrieval="agent", agent_llm=llm)

    assert result["retrieval_used"] == "agent"
    assert len(result["retrieval_log"]) == 2
    genres = {t.genres[0] for t in result["selected_tracks"]}
    assert genres <= {"Jazz", "Folk"}
    assert len(result["selected_tracks"]) == 10


def test_graph_agent_failure_falls_back(catalog, offline_llm_down):
    class Broken(ScriptedToolLLM):
        def invoke(self, *a, **k):
            raise RuntimeError("model does not support tools")

    result = run_curation("folk", 10, catalog, retrieval="agent", agent_llm=Broken(responses=[]))
    assert result["retrieval_used"] == "heuristic"
    assert len(result["selected_tracks"]) == 10


def test_graph_top_tracks_mode(catalog, offline_llm_down):
    result = run_curation("my top tracks", 5, catalog)
    assert result["retrieval_used"] == "taste"
    assert [t.artists[0] for t in result["selected_tracks"]] == ["Gravel Kings"] * 5
