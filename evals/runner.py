"""Run curation configs over the gold prompt set and record everything.

    python -m app eval run --configs baseline,llm_rank,agent --name dev --limit 12

Design choices:
* One intent parse per prompt, shared by every config, so differences between
  configs come from retrieval/ranking/sequencing and not from parser variance.
* Results are written after every (prompt, config) pair and a rerun with the
  same --name resumes, so long local-LLM runs survive interruptions.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from app.agent.ranker import RankWeights
from core.logging import logger
from evals.metrics import playlist_metrics

EVALS_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = EVALS_DIR / "prompts.yaml"
RESULTS_DIR = EVALS_DIR / "results"


@dataclass(frozen=True)
class EvalConfig:
    """One system variant to evaluate."""

    name: str
    description: str
    retrieval: str = "heuristic"          # heuristic | agent
    use_stage2: bool = True
    sequence: bool = False
    weights: Dict[str, float] = field(default_factory=dict)  # RankWeights overrides
    # Post-process another config's playlists instead of re-running the pipeline,
    # so the effect of `sequence` is isolated from run-to-run LLM variance.
    derive_from: Optional[str] = None

    def rank_weights(self) -> Optional[RankWeights]:
        return RankWeights(**self.weights) if self.weights else None


CONFIGS: Dict[str, EvalConfig] = {c.name: c for c in [
    EvalConfig("baseline", "Keyword retrieval, heuristic ranking (no LLM after intent parsing)",
               retrieval="heuristic", use_stage2=False),
    EvalConfig("baseline_fit", "Baseline with stage-1 weight shifted from taste to feature/era fit",
               retrieval="heuristic", use_stage2=False, weights={"fit": 0.6, "taste": 0.2, "base": 0.2}),
    EvalConfig("baseline_adaptive", "Baseline; taste weight only for requests that ask for personal taste",
               retrieval="heuristic", use_stage2=False, weights={"taste_when_impersonal": 0.1}),
    EvalConfig("llm_rank", "Keyword retrieval + batched LLM relevance scoring",
               retrieval="heuristic"),
    EvalConfig("agent", "Tool-calling retrieval agent + LLM relevance scoring",
               retrieval="agent"),
    EvalConfig("agent_seq", "The agent config's playlists re-ordered along the intent's energy arc",
               retrieval="agent", sequence=True, derive_from="agent"),
    EvalConfig("agent_fit", "Agent + LLM scoring, stage-1 weights shifted from taste to feature fit",
               retrieval="agent", weights={"fit": 0.6, "taste": 0.2, "base": 0.2}),
    EvalConfig("agent_adaptive", "Agent + LLM scoring with adaptive taste weight",
               retrieval="agent", weights={"taste_when_impersonal": 0.1}),
]}


def load_prompts(path: Path = PROMPTS_PATH) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    prompts = [{**defaults, **p} for p in data["prompts"]]
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prompt ids in prompts.yaml")
    return prompts


def select_prompts(prompts, ids=None, categories=None, limit=None, stratified=True):
    """Filter prompts; with a limit, take round-robin across categories (stratified)."""
    if ids:
        prompts = [p for p in prompts if p["id"] in set(ids)]
    if categories:
        prompts = [p for p in prompts if p["category"] in set(categories)]
    if not limit or limit >= len(prompts):
        return prompts
    if not stratified:
        return prompts[:limit]
    by_cat: Dict[str, List[Dict]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    picked: List[Dict] = []
    while len(picked) < limit:
        for group in by_cat.values():
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    return picked


def track_record(t) -> Dict[str, Any]:
    return t.model_dump(include={"id", "uri", "name", "artists", "genres", "year", "popularity",
                                 "audio_features", "why"})


class EvalRunner:
    def __init__(
        self,
        catalog_factory: Callable[[str], Any],
        judge=None,
        parse_intent: Optional[Callable] = None,
        agent_llm=None,
    ):
        """
        Args:
            catalog_factory: profile name -> CatalogProvider (cached per profile).
            judge: PlaylistJudge or None to skip judging.
            parse_intent: override for intent parsing (tests).
            agent_llm: override chat model for the retrieval agent (tests).
        """
        self._catalogs: Dict[str, Any] = {}
        self.catalog_factory = catalog_factory
        self.judge = judge
        self.agent_llm = agent_llm
        if parse_intent is None:
            from app.agent.intent import parse_intent as default_parse
            parse_intent = default_parse
        self.parse_intent = parse_intent

    def catalog(self, profile: str):
        if profile not in self._catalogs:
            self._catalogs[profile] = self.catalog_factory(profile)
        return self._catalogs[profile]

    def run(
        self,
        prompts: List[Dict[str, Any]],
        configs: List[EvalConfig],
        name: str,
        out_dir: Path = RESULTS_DIR,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from app.agent.graph import run_curation
        from app.agent.intent import IntentNormalized

        names = [c.name for c in configs]
        for c in configs:
            if c.derive_from and (c.derive_from not in names or names.index(c.derive_from) > names.index(c.name)):
                raise ValueError(f"Config '{c.name}' derives from '{c.derive_from}', which must run before it")

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.json"
        results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "run": name, "started": datetime.now().isoformat(timespec="seconds"),
            "meta": meta or {}, "configs": {}, "intents": {}, "records": [],
        }
        for c in configs:
            results["configs"][c.name] = asdict(c)
        done = {(r["prompt_id"], r["config"]) for r in results["records"]}

        total = len(prompts) * len(configs)
        step = 0
        for prompt in prompts:
            catalog = self.catalog(prompt["profile"])
            seeds = sorted(catalog.top_artists())

            if prompt["id"] in results["intents"]:
                intent = IntentNormalized(**results["intents"][prompt["id"]])
            else:
                t0 = time.perf_counter()
                intent = self.parse_intent(prompt["prompt"], prompt["length"])
                results["intents"][prompt["id"]] = {**intent.model_dump(), "parse_s": round(time.perf_counter() - t0, 2)}

            for config in configs:
                step += 1
                if (prompt["id"], config.name) in done:
                    continue
                logger.info(f"[eval] ({step}/{total}) {prompt['id']} x {config.name}")
                if config.derive_from:
                    source = next(r for r in results["records"]
                                  if r["prompt_id"] == prompt["id"] and r["config"] == config.derive_from)
                    record = self._derive(source, config, intent, prompt, seeds)
                    results["records"].append(record)
                    path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
                    continue
                t0 = time.perf_counter()
                error = None
                try:
                    state = run_curation(
                        prompt["prompt"], prompt["length"], catalog,
                        retrieval=config.retrieval, weights=config.rank_weights(),
                        agent_llm=self.agent_llm, use_stage2=config.use_stage2,
                        sequence=config.sequence, intent=intent.model_copy(deep=True),
                    )
                    tracks = state.get("selected_tracks", [])
                except Exception as e:  # record the failure, keep the run going
                    logger.error(f"[eval] {prompt['id']} x {config.name} failed: {e}")
                    state, tracks, error = {}, [], str(e)
                curate_s = time.perf_counter() - t0

                record = {
                    "prompt_id": prompt["id"], "category": prompt["category"], "prompt": prompt["prompt"],
                    "profile": prompt["profile"], "config": config.name, "error": error,
                    "retrieval_used": state.get("retrieval_used"),
                    "tool_calls": [{k: v for k, v in c.items() if k not in ("call_id", "ids")}
                                   for c in state.get("retrieval_log") or []],
                    "tracks": [track_record(t) for t in tracks],
                    "metrics": playlist_metrics(tracks, prompt, prompt["length"], seeds),
                    "timing": {"curate_s": round(curate_s, 2)},
                }
                if self.judge is not None:
                    t1 = time.perf_counter()
                    try:
                        record["judge"] = self.judge.judge(prompt["prompt"], tracks, seeds)
                    except Exception as e:
                        logger.error(f"[eval] judge failed for {prompt['id']} x {config.name}: {e}")
                        record["judge"] = None
                    record["timing"]["judge_s"] = round(time.perf_counter() - t1, 2)

                results["records"].append(record)
                results["updated"] = datetime.now().isoformat(timespec="seconds")
                path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

        return results

    @staticmethod
    def _derive(source: Dict[str, Any], config: EvalConfig, intent, prompt: Dict[str, Any], seeds) -> Dict[str, Any]:
        """Re-sequence a source config's playlist; membership (and judge labels) unchanged."""
        from app.agent.sequencer import sequence_by_energy
        from app.core.draft_models import TrackRef

        tracks = [TrackRef(**t) for t in source["tracks"]]
        if config.sequence:
            original = [t.id for t in tracks]
            tracks = sequence_by_energy(tracks, (intent.structure or {}).get("phases", []))
            order = [original.index(t.id) for t in tracks]
        else:
            order = list(range(len(tracks)))
        judge = source.get("judge")
        if judge and judge.get("track_fits"):
            judge = {**judge, "track_fits": [judge["track_fits"][i] for i in order]}
        return {
            **{k: source[k] for k in ("prompt_id", "category", "prompt", "profile", "error",
                                      "retrieval_used", "tool_calls")},
            "config": config.name, "derived_from": config.derive_from,
            "tracks": [track_record(t) for t in tracks],
            "metrics": playlist_metrics(tracks, prompt, prompt["length"], seeds),
            "judge": judge, "timing": dict(source.get("timing") or {}),
        }
