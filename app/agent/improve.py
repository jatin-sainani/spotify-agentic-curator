"""Feedback-driven draft improvement with a human-in-the-loop checkpoint.

    propose ──► review (interrupt: a human accepts/rejects each swap) ──► apply

propose scores the current tracks and fresh candidates against the original
request plus the feedback, then pairs the weakest current tracks with stronger
replacements. review pauses the graph with LangGraph's interrupt(); the caller
resumes it with one decision per proposal. apply performs only accepted swaps
(in place, keeping positions) and logs every decision as a feedback event.
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.agent.draft_ops import _event
from app.core.draft_models import Draft, TrackRef
from core.logging import logger

DEFAULT_MAX_SWAPS = 3
MIN_GAIN = 0.05       # replacement must beat the current track by at least this score...
MIN_NEW_SCORE = 0.5   # ...and be a good fit in its own right (not just less bad)


class ImproveState(TypedDict, total=False):
    draft: Dict[str, Any]
    feedback: str
    proposals: List[Dict[str, Any]]
    decisions: List[bool]
    applied: int


def _score_all(tracks: List[TrackRef], intent, top_artists, use_llm: bool, request: str) -> Dict[str, float]:
    from app.agent.ranker import _stage1_pre_score, _stage2_llm_score

    stage1 = _stage1_pre_score(tracks, intent, top_artists)
    if not use_llm:
        return {t.id: s for t, s in stage1}
    return {t.id: s for t, s, _ in _stage2_llm_score(stage1, intent, request_text=request)}


def propose_swaps(
    draft: Draft,
    feedback: str,
    catalog,
    max_swaps: int = DEFAULT_MAX_SWAPS,
    use_llm: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Pair the weakest current tracks with better-fitting new candidates."""
    from app.agent.candidates import build_candidates
    from app.agent.intent import parse_intent
    from app.agent.ranker import _llm_available, song_key

    request = f"{draft.intent_text}. Listener feedback: {feedback}"
    intent = parse_intent(request, draft.length)
    if use_llm is None:
        use_llm = _llm_available()

    # Exclude songs already in the draft (any recording of them), and dedupe the pool.
    taken = {song_key(t) for t in draft.tracks}
    pool = []
    # Search only: feedback asks for something different, so don't refill from taste.
    for t in build_candidates(catalog, limit=60, intent=intent, intent_text=request, taste_share=0.0):
        key = song_key(t)
        if key not in taken:
            taken.add(key)
            pool.append(t)
    from app.agent.safety import filter_flagged

    pool = filter_flagged(pool, "improve")
    if not pool:
        return []

    top_artists = catalog.top_artists()
    # Score current tracks and candidates in one pass so scores are comparable.
    scores = _score_all(list(draft.tracks) + pool[:30], intent, top_artists, use_llm, request)
    weakest = sorted(draft.tracks, key=lambda t: scores.get(t.id, 0.0))
    best_new = sorted(pool[:30], key=lambda t: scores.get(t.id, 0.0), reverse=True)

    proposals = []
    for old, new in zip(weakest, best_new):
        if len(proposals) >= max_swaps:
            break
        new_score = scores.get(new.id, 0.0)
        if new_score < MIN_NEW_SCORE or new_score - scores.get(old.id, 0.0) < MIN_GAIN:
            break  # both lists are sorted, so no later pair does better
        proposals.append({
            "remove": old.model_dump(),
            "add": new.model_dump(),
            "old_score": round(scores.get(old.id, 0.0), 3),
            "new_score": round(scores.get(new.id, 0.0), 3),
        })
    logger.info(f"[improve] {len(proposals)} swap proposals for feedback {feedback!r} (llm={use_llm})")
    return proposals


def apply_swaps(draft: Draft, proposals: List[Dict[str, Any]], decisions: List[bool], feedback: str):
    """Apply accepted swaps in place (same positions); log every decision. Returns (draft, applied)."""
    applied = 0
    for proposal, accepted in zip(proposals, decisions):
        old_id = proposal["remove"]["id"]
        data = {"feedback": feedback, "removed": old_id, "added": proposal["add"]["id"]}
        if not accepted:
            _event(draft, "swap_rejected", **data)
            continue
        idx = next((i for i, t in enumerate(draft.tracks) if t.id == old_id), None)
        if idx is None:
            continue
        new = TrackRef(**proposal["add"])
        new.why = f"Swapped in for feedback: {feedback}"
        draft.tracks[idx] = new
        _event(draft, "swap_accepted", **data)
        applied += 1
    return draft, applied


def build_improve_graph(catalog, max_swaps: int = DEFAULT_MAX_SWAPS, use_llm: Optional[bool] = None):
    def propose_node(state: ImproveState) -> dict:
        draft = Draft(**state["draft"])
        return {"proposals": propose_swaps(draft, state["feedback"], catalog, max_swaps, use_llm)}

    def review_node(state: ImproveState) -> dict:
        if not state["proposals"]:
            return {"decisions": []}
        decisions = interrupt({"proposals": state["proposals"]})
        if not isinstance(decisions, list) or len(decisions) != len(state["proposals"]):
            raise ValueError("Resume with one boolean decision per proposal")
        return {"decisions": [bool(d) for d in decisions]}

    def apply_node(state: ImproveState) -> dict:
        draft, applied = apply_swaps(Draft(**state["draft"]), state["proposals"], state["decisions"],
                                     state["feedback"])
        return {"draft": draft.model_dump(mode="json"), "applied": applied}

    graph = StateGraph(ImproveState)
    graph.add_node("propose", propose_node)
    graph.add_node("review", review_node)
    graph.add_node("apply", apply_node)
    graph.set_entry_point("propose")
    graph.add_edge("propose", "review")
    graph.add_edge("review", "apply")
    graph.add_edge("apply", END)
    return graph.compile(checkpointer=MemorySaver())


def improve_draft(
    draft: Draft,
    feedback: str,
    catalog,
    decide: Callable[[List[Dict[str, Any]]], List[bool]],
    max_swaps: int = DEFAULT_MAX_SWAPS,
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run propose → (pause for decide) → apply. Returns {"draft", "proposals", "applied"}."""
    graph = build_improve_graph(catalog, max_swaps=max_swaps, use_llm=use_llm)
    config = {"configurable": {"thread_id": f"improve-{draft.id}"}}

    state = graph.invoke({"draft": draft.model_dump(mode="json"), "feedback": feedback}, config)
    pending = state.get("__interrupt__")
    if pending:
        proposals = pending[0].value["proposals"]
        state = graph.invoke(Command(resume=decide(proposals)), config)

    return {
        "draft": Draft(**state["draft"]),
        "proposals": state.get("proposals", []),
        "applied": state.get("applied", 0),
    }
