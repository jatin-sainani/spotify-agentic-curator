"""LLM-as-judge for generated playlists.

Two kinds of labels per playlist:

* Per-track relevance on a graded 0 / 0.5 / 1 scale (Cranfield-style labels),
  which gives judge precision and can be validated against human labels.
* List-level rubric dimensions, each 0 / 0.5 / 1:
    intent_alignment  does the playlist deliver what was asked (incl. exclusions)?
    cohesion          do the tracks belong together as one listening experience?
    diversity         varied artists/songs without drifting off-brief?
    personalization   does it fit the listener's taste where the request calls for it?
    appropriateness   free of jarring or off-brief picks (wrong season, duplicates...)?

The judge sees what a human rater would see (title, artists, genres, year), not
audio-feature numbers, so its labels are comparable with human labels.

Bias controls: pass 1 shows tracks in playlist order, pass 2 in a seeded shuffle;
labels are mapped back and averaged. Pass-to-pass agreement is reported as the
judge's self-consistency. All rubric dimensions are order-independent on purpose;
sequencing is measured deterministically (evals.metrics arc_score).
"""

import random
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.draft_models import TrackRef
from core.logging import logger

DIMENSIONS = ("intent_alignment", "cohesion", "diversity", "personalization", "appropriateness")
GRADES = (0.0, 0.5, 1.0)

JUDGE_SYSTEM_PROMPT = """\
You are an expert music curator grading a playlist that a system generated for a \
listener's request. Be strict and specific; a track only earns 1 if a thoughtful \
human curator would happily include it for this exact request.

Grade each track's fit to the request:
  1   = clearly fits the request
  0.5 = partially fits (right genre but wrong mood/era, or vice versa)
  0   = does not fit, or violates an explicit exclusion in the request

Then grade the whole playlist on each dimension (0, 0.5 or 1):
  intent_alignment: delivers what was asked overall, including any exclusions
  cohesion: the tracks belong together as one listening experience
  diversity: varied artists and songs without drifting off the brief
  personalization: fits the listener's taste where the request asks for it \
(1 if the request is not about personal taste and nothing clashes with it)
  appropriateness: no jarring or off-brief picks (e.g. holiday songs, duplicates, \
novelty tracks, explicit aggression when asked for calm)

{untrusted_notice}
Respond with ONLY valid JSON:
{{"tracks": [{{"n": <number>, "fit": <0|0.5|1>}}, ...],
  "intent_alignment": <0|0.5|1>, "cohesion": <0|0.5|1>, "diversity": <0|0.5|1>,
  "personalization": <0|0.5|1>, "appropriateness": <0|0.5|1>,
  "summary": "<one sentence on the biggest problem, or 'none'>"}}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_PROMPT),
    ("human", "Listener request: {request}\nListener's favorite artists: {taste}\n\n"
              "Playlist:\n{tracks_block}"),
])


def judge_track_line(n: int, track: TrackRef) -> str:
    """What a human rater sees for a track."""
    from app.agent.safety import for_prompt

    track = for_prompt(track)
    specific = track.genres[1:4] or track.genres[:1]
    bits = [f"{n}. \"{track.name}\" by {', '.join(track.artists) or 'Unknown'}"]
    if specific:
        bits.append(f"genres: {', '.join(specific)}")
    if track.year:
        bits.append(str(track.year))
    return " | ".join(bits)


def snap(value: Any) -> Optional[float]:
    """Coerce a label to the nearest grade (0, 0.5, 1); None if unparseable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(GRADES, key=lambda g: abs(g - max(0.0, min(1.0, v))))


class PlaylistJudge:
    """Grades playlists with an LLM. Pass `llm` to inject a model (tests)."""

    def __init__(self, llm=None, passes: int = 2, seed: int = 13):
        if llm is None:
            from app.agent.llm import get_llm

            llm = get_llm(temperature=0.0, role="judge")
        self.chain = JUDGE_PROMPT | llm | JsonOutputParser()
        self.passes = max(1, passes)
        self.seed = seed

    def _one_pass(self, request: str, taste: str, tracks: List[TrackRef], order: List[int]) -> Dict[str, Any]:
        from app.agent.safety import untrusted_notice, wrap_catalog_data

        block = "\n".join(judge_track_line(i + 1, tracks[j]) for i, j in enumerate(order))
        raw = self.chain.invoke({"request": request, "taste": taste, "tracks_block": wrap_catalog_data(block),
                                 "untrusted_notice": untrusted_notice()})
        raw = raw if isinstance(raw, dict) else {}

        fits: List[Optional[float]] = [None] * len(tracks)
        for item in raw.get("tracks", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                shown = int(item.get("n")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= shown < len(order):
                fits[order[shown]] = snap(item.get("fit"))
        dims = {d: snap(raw.get(d)) for d in DIMENSIONS}
        return {"fits": fits, "dims": dims, "summary": str(raw.get("summary", ""))[:300]}

    def judge(self, request: str, tracks: List[TrackRef], taste_artists: List[str]) -> Dict[str, Any]:
        """Grade one playlist. Returns averaged labels plus per-pass detail."""
        if not tracks:
            return {"track_fits": [], "dims": {d: 0.0 for d in DIMENSIONS}, "precision": 0.0,
                    "self_agreement": None, "summaries": [], "failed_passes": 0}

        taste = ", ".join(taste_artists) or "unknown"
        rng = random.Random(f"{self.seed}:{request}")
        passes, failed = [], 0
        for p in range(self.passes):
            order = list(range(len(tracks)))
            if p > 0:
                rng.shuffle(order)
            try:
                passes.append(self._one_pass(request, taste, tracks, order))
            except Exception as e:  # malformed JSON, timeouts: drop the pass, keep going
                failed += 1
                logger.warning(f"[judge] pass {p + 1} failed: {e}")

        def mean(values):
            vals = [v for v in values if v is not None]
            return sum(vals) / len(vals) if vals else None

        track_fits = [mean([ps["fits"][i] for ps in passes]) for i in range(len(tracks))]
        dims = {d: mean([ps["dims"][d] for ps in passes]) for d in DIMENSIONS}
        known = [f for f in track_fits if f is not None]

        agreement = None
        if len(passes) >= 2:
            pairs = [(a, b) for a, b in zip(passes[0]["fits"], passes[1]["fits"]) if a is not None and b is not None]
            agreement = sum(a == b for a, b in pairs) / len(pairs) if pairs else None

        return {
            "track_fits": track_fits,
            "dims": dims,
            "precision": sum(known) / len(known) if known else None,
            "self_agreement": agreement,
            "summaries": [ps["summary"] for ps in passes],
            "failed_passes": failed,
        }
