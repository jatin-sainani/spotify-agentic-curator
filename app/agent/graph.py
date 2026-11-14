"""LangGraph StateGraph for agentic playlist curation.

  parse_intent ─┬─(top_tracks)──► taste_tracks ──────────────────────────► END
                └─(curate)──► retrieve ─► stage1_score ─┬─► stage2_score ─► select_tracks ─► END
                                                        └─(LLM unavailable)──┘

retrieve runs the tool-calling retrieval agent (app/agent/retrieval.py) when
the configured LLM is reachable, topping up with heuristic catalog search; it
falls back to heuristic-only retrieval if the agent fails. Stage 2 is likewise
skipped when the LLM is unreachable, so the pipeline always returns a playlist.
"""

from typing import Any, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from core.logging import logger

Retrieval = Literal["auto", "agent", "heuristic"]


class CurationState(TypedDict, total=False):
    """Typed state flowing through the curation graph."""

    # -- inputs --
    intent_text: str
    length: int
    retrieval: str             # "auto" | "agent" | "heuristic"
    use_stage2: bool           # False forces heuristic-only ranking (eval baseline)
    sequence: bool             # order the playlist along the intent's energy arc

    # -- after parse_intent --
    intent: Optional[Any]      # IntentNormalized

    # -- after retrieve --
    candidates: list           # List[TrackRef]
    top_artists: set           # Set[str]
    retrieval_used: str        # what actually ran: "agent" | "heuristic"
    retrieval_log: list        # tool calls made by the agent

    # -- after stage1_score / stage2_score --
    stage1_results: list       # List[Tuple[TrackRef, float]], all candidates, sorted
    stage2_results: list       # List[Tuple[TrackRef, float, dict]]

    # -- output --
    selected_tracks: list      # List[TrackRef]


def _llm_up() -> bool:
    from app.agent.llm import provider_available

    return provider_available()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_intent_node(state: CurationState) -> dict:
    """Parse natural-language intent into the normalized schema."""
    from app.agent.intent import parse_intent

    if state.get("intent") is not None:  # preset (e.g. evals share one parse across configs)
        return {}
    intent = parse_intent(state["intent_text"], state["length"])
    logger.info(f"[graph] Intent parsed: mode={intent.mode}")
    return {"intent": intent}


def route_after_intent(state: CurationState) -> str:
    return "taste_tracks" if state["intent"].mode == "top_tracks" else "retrieve"


def make_taste_tracks_node(catalog):
    def taste_tracks_node(state: CurationState) -> dict:
        tracks = catalog.get_user_taste(limit=state["length"])
        for t in tracks:
            t.why = "From your top tracks"
        logger.info(f"[graph] top_tracks mode: {len(tracks)} tracks from listener taste")
        return {"selected_tracks": tracks, "retrieval_used": "taste"}

    return taste_tracks_node


def make_retrieve_node(catalog, agent_llm=None):
    """Bind the catalog (and optionally a pre-built agent LLM) into the retrieve node.

    The catalog is injected via closure rather than graph state: LangGraph only
    carries keys declared in CurationState, and a live client is not state.
    """
    from app.agent.candidates import build_candidates
    from app.agent.retrieval import agent_retrieve, merge_pools

    def retrieve_node(state: CurationState) -> dict:
        length = state["length"]
        limit = min(200, length * 3)
        mode = state.get("retrieval", "auto")
        intent = state["intent"]

        heuristic = lambda: build_candidates(  # noqa: E731
            catalog, limit=limit, intent=intent, intent_text=state["intent_text"]
        )

        # "agent" and "auto" both need a reachable model; an injected agent_llm counts as one.
        use_agent = mode in ("agent", "auto") and (agent_llm is not None or _llm_up())
        if mode == "agent" and not use_agent:
            logger.warning("[graph] Agent retrieval requested but the LLM is unreachable; using heuristic search")
        if use_agent:
            try:
                session = agent_retrieve(
                    catalog, state["intent_text"], intent.description, target=limit, llm=agent_llm
                )
                agent_pool = list(session.pool.values())
                # Top up thin agent pools so ranking always has enough to choose from.
                pool = agent_pool if len(agent_pool) >= limit // 2 else merge_pools(agent_pool, heuristic(), limit)
                logger.info(f"[graph] Agent retrieval: {len(agent_pool)} found, {len(pool)} candidates after top-up")
                return {
                    "candidates": pool[:limit],
                    "top_artists": catalog.top_artists(),
                    "retrieval_used": "agent",
                    "retrieval_log": session.calls,
                }
            except Exception as e:
                logger.warning(f"[graph] Agent retrieval failed ({e}); falling back to heuristic search")

        pool = heuristic()
        logger.info(f"[graph] Heuristic retrieval: {len(pool)} candidates from '{catalog.name}' catalog")
        return {
            "candidates": pool,
            "top_artists": catalog.top_artists(),
            "retrieval_used": "heuristic",
            "retrieval_log": [],
        }

    return retrieve_node


def make_stage1_node(weights):
    def stage1_score_node(state: CurationState) -> dict:
        """Stage 1: heuristic pre-scoring (audio-feature fit + taste similarity)."""
        from app.agent.ranker import _stage1_pre_score
        from app.agent.safety import filter_flagged

        candidates = filter_flagged(state["candidates"], "ranking")
        results = _stage1_pre_score(candidates, state["intent"], state["top_artists"], weights)
        logger.info(f"[graph] Stage 1 scored {len(results)} candidates")
        return {"stage1_results": results}

    return stage1_score_node


def make_stage2_node(weights):
    def stage2_score_node(state: CurationState) -> dict:
        """Stage 2: batched LLM semantic scoring of the stage-1 top-K."""
        from app.agent.ranker import STAGE2_TOP_K, _stage2_llm_score, heuristic_only

        top = state["stage1_results"][:STAGE2_TOP_K]
        rest = state["stage1_results"][STAGE2_TOP_K:]
        scored = _stage2_llm_score(top, state["intent"], weights=weights, request_text=state["intent_text"])
        # Unscored tail stays available so diversity caps can't leave the playlist short;
        # it ranks below every LLM-scored track.
        tail = [(t, s * 0.01, sem) for t, s, sem in heuristic_only(rest)]
        return {"stage2_results": scored + tail}

    return stage2_score_node


def select_tracks_node(state: CurationState) -> dict:
    """Diversity-aware selection plus per-track explanations."""
    from app.agent.ranker import _generate_semantic_selection_reason, heuristic_only, select_with_diversity

    scored = state.get("stage2_results") or heuristic_only(state["stage1_results"])
    picked = select_with_diversity(scored, state["length"])
    selected = []
    for track, final_score, semantic in picked:
        track.why = _generate_semantic_selection_reason(
            track, final_score, semantic, state["intent"], state.get("top_artists")
        )
        selected.append(track)
    if state.get("sequence"):
        from app.agent.sequencer import sequence_by_energy

        selected = sequence_by_energy(selected, (state["intent"].structure or {}).get("phases", []))
    logger.info(f"[graph] Selected {len(selected)}/{state['length']} tracks")
    return {"selected_tracks": selected}


def route_after_stage1(state: CurationState) -> str:
    """Conditional edge: LLM scoring if the provider is reachable, else straight to selection."""
    from app.agent.llm import model_name

    if state.get("use_stage2") is False:
        logger.info("[graph] Stage 2 disabled for this run -> heuristic-only ranking")
        return "select_tracks"
    if _llm_up():
        logger.info(f"[graph] LLM available ({model_name()}) -> Stage 2 (semantic scoring)")
        return "stage2_score"
    logger.warning(f"[graph] LLM unavailable ({model_name()}) -> skipping Stage 2 (graceful degradation)")
    return "select_tracks"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def create_curation_graph(catalog, weights=None, agent_llm=None):
    """Assemble and compile the curation graph.

    Args:
        catalog: CatalogProvider to retrieve from.
        weights: Optional RankWeights (defaults to the baseline weights).
        agent_llm: Optional pre-built chat model for the retrieval agent (tests/evals).
    """
    from app.agent.ranker import DEFAULT_WEIGHTS

    weights = weights or DEFAULT_WEIGHTS
    graph = StateGraph(CurationState)

    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("taste_tracks", make_taste_tracks_node(catalog))
    graph.add_node("retrieve", make_retrieve_node(catalog, agent_llm))
    graph.add_node("stage1_score", make_stage1_node(weights))
    graph.add_node("stage2_score", make_stage2_node(weights))
    graph.add_node("select_tracks", select_tracks_node)

    graph.set_entry_point("parse_intent")
    graph.add_conditional_edges("parse_intent", route_after_intent,
                                {"taste_tracks": "taste_tracks", "retrieve": "retrieve"})
    graph.add_edge("taste_tracks", END)
    graph.add_edge("retrieve", "stage1_score")
    graph.add_conditional_edges("stage1_score", route_after_stage1,
                                {"stage2_score": "stage2_score", "select_tracks": "select_tracks"})
    graph.add_edge("stage2_score", "select_tracks")
    graph.add_edge("select_tracks", END)
    return graph.compile()


def run_curation(
    intent_text: str,
    length: int,
    catalog,
    retrieval: Retrieval = "auto",
    weights=None,
    agent_llm=None,
    use_stage2: bool = True,
    sequence: bool = False,
    intent=None,
) -> CurationState:
    """Run the curation graph and return the full final state (tracks + diagnostics).

    Args:
        retrieval: "agent", "heuristic" or "auto".
        weights: RankWeights override.
        agent_llm: Chat model for the retrieval agent (tests/evals).
        use_stage2: False skips LLM scoring even if the provider is up.
        sequence: Reorder the final playlist along the intent's energy arc.
        intent: A pre-parsed IntentNormalized (skips parse_intent).
    """
    graph = create_curation_graph(catalog, weights=weights, agent_llm=agent_llm)
    logger.info(f"[graph] Starting curation: '{intent_text}' (length={length}, retrieval={retrieval})")
    state = {"intent_text": intent_text, "length": length, "retrieval": retrieval,
             "use_stage2": use_stage2, "sequence": sequence}
    if intent is not None:
        state["intent"] = intent
    result = graph.invoke(state)
    logger.info(f"[graph] Complete: {len(result.get('selected_tracks', []))} tracks "
                f"(retrieval={result.get('retrieval_used')})")
    return result


def run_curation_graph(intent_text: str, length: int, catalog, retrieval: Retrieval = "auto") -> List:
    """Run the curation graph and return only the selected tracks."""
    return run_curation(intent_text, length, catalog, retrieval=retrieval).get("selected_tracks", [])
