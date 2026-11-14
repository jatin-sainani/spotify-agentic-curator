"""Tests for the evaluation harness (no real LLM anywhere)."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent.intent import IntentNormalized
from app.agent.sequencer import sequence_by_energy, slot_targets
from app.catalog.offline import OfflineCatalog
from app.core.draft_models import TrackRef
from evals import agreement, metrics
from evals.judge import PlaylistJudge, snap
from evals.report import build_report, paired_diff
from evals.runner import CONFIGS, EvalRunner, load_prompts, select_prompts

FIXTURE = Path(__file__).parent / "fixtures" / "catalog_synthetic.csv"


def t(tid, artist="A", genres=("Folk", "indie folk"), year=2010, name=None, **features):
    return TrackRef(uri=f"spotify:track:{tid}", id=str(tid), name=name or f"Song {tid}", artists=[artist],
                    genres=list(genres), year=year, audio_features=features or None)


# -- statistics ----------------------------------------------------------------

def test_spearman_and_ties():
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert metrics.spearman([1, 1, 1], [1, 2, 3]) is None
    assert metrics._ranks([5, 1, 5]) == [2.5, 1.0, 2.5]


def test_weighted_kappa_known_values():
    assert agreement.weighted_kappa([0, 0.5, 1, 1], [0, 0.5, 1, 1]) == pytest.approx(1.0)
    # Systematic disagreement in opposite directions is worse than chance.
    assert agreement.weighted_kappa([0, 1, 0, 1], [1, 0, 1, 0]) < 0
    # Off-by-one-grade costs less than off-by-two.
    near = agreement.weighted_kappa([0, 0.5, 1, 0.5, 1, 0], [0.5, 0.5, 1, 0.5, 0.5, 0])
    far = agreement.weighted_kappa([0, 0.5, 1, 0.5, 1, 0], [1, 0.5, 1, 0.5, 0, 0])
    assert near > far


def test_entropy_and_arc_curves():
    assert metrics.normalized_entropy(["a", "a"]) == 0.0
    assert metrics.normalized_entropy(["a", "b", "c"]) == pytest.approx(1.0)
    assert metrics.arc_curve("build", 3) == [0.0, 0.5, 1.0]
    assert metrics.arc_curve("peak", 3) == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        metrics.arc_curve("zigzag", 3)


# -- playlist metrics --------------------------------------------------------------

def test_playlist_metrics_against_gold():
    tracks = [
        t(1, "X", ("Rock", "new wave"), 1984, energy=0.8, valence=0.7),
        t(2, "Y", ("Pop", "synthpop"), 1986, energy=0.6, valence=0.9),
        t(3, "X", ("Pop", "dance pop"), 2015, name="Last Christmas", energy=0.3, valence=0.2),
    ]
    gold = {"genres_any": ["new wave", "synthpop"], "era": [1980, 1989], "avoid": ["christmas"],
            "targets": {"energy": [0.5, 1.0]}, "arc": "build"}
    m = metrics.playlist_metrics(tracks, gold, length=4, seed_artists=["Y"])

    assert m["fill_rate"] == 0.75
    assert m["genre_hit_rate"] == pytest.approx(2 / 3)
    assert m["era_hit_rate"] == pytest.approx(2 / 3)
    assert m["avoid_clean_rate"] == pytest.approx(2 / 3)
    assert m["target_hit_rate"] == pytest.approx(2 / 3)
    assert 2 / 3 < m["target_fit"] < 1  # 0.3 is 0.2 below the range: partial credit
    assert m["arc_score"] == pytest.approx(-1.0)  # energy goes 0.8 -> 0.6 -> 0.3, the opposite of "build"
    assert m["artist_diversity"] == pytest.approx(2 / 3)
    assert m["familiarity"] == pytest.approx(1 / 3)
    expected_gold = (2 / 3 + 2 / 3 + m["target_fit"] + 2 / 3 + 0.0) / 5
    assert m["gold_score"] == pytest.approx(expected_gold)


def test_metrics_skip_undefined_expectations_and_empty():
    m = metrics.playlist_metrics([t(1, energy=0.5)], {}, length=1)
    assert m["gold_score"] is None and "genre_hit_rate" not in m
    assert metrics.playlist_metrics([], {"genres_any": ["x"]}, length=5)["gold_score"] == 0.0


def test_duplicate_songs_counted():
    m = metrics.playlist_metrics([t(1, name="Blue Monday"), t(2, name="Blue Monday - 2015 Remaster")], {}, 2)
    assert m["duplicate_songs"] == 1.0


# -- sequencer ----------------------------------------------------------------------

PHASES = [{"name": "warmup", "pct": 0.2, "energy_target": 0.4},
          {"name": "cruise", "pct": 0.6, "energy_target": 0.8},
          {"name": "landing", "pct": 0.2, "energy_target": 0.3}]


def test_slot_targets_sum_to_n():
    assert slot_targets(PHASES, 10) == [0.4] * 2 + [0.8] * 6 + [0.3] * 2
    assert len(slot_targets(PHASES, 7)) == 7


def test_sequence_follows_arc_and_improves_arc_score():
    tracks = [t(i, energy=e) for i, e in enumerate([0.9, 0.2, 0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.95, 0.1])]
    ordered = sequence_by_energy(tracks, PHASES)
    energies = [x.audio_features["energy"] for x in ordered]
    assert sorted(energies) == sorted(x.audio_features["energy"] for x in tracks)
    assert max(energies[2:8]) == 0.95 and min(energies[2:8]) > max(energies[:2] + energies[8:])
    before = metrics.playlist_metrics(tracks, {"arc": "peak"}, 10)["arc_score"]
    after = metrics.playlist_metrics(ordered, {"arc": "peak"}, 10)["arc_score"]
    assert after > before


def test_sequence_leaves_flat_or_featureless_lists_alone():
    tracks = [t(1, energy=0.5), t(2), t(3, energy=0.2), t(4, energy=0.9), t(5, energy=0.7), t(6, energy=0.3)]
    assert sequence_by_energy(tracks, [{"pct": 1.0, "energy_target": 0.5}]) == tracks
    ordered = sequence_by_energy(tracks, PHASES)
    assert ordered[-1].id == "2"  # unknown energy goes last
    assert len(ordered) == 6
    # Two tracks under a 20/60/20 split round to one flat target: nothing to sequence.
    assert sequence_by_energy(tracks[:2], PHASES) == tracks[:2]


# -- judge -------------------------------------------------------------------------

def fake_judge_llm(responses):
    return FakeListChatModel(responses=[json.dumps(r) for r in responses])


def test_judge_maps_shuffled_labels_back_and_averages():
    tracks = [t(i, name=f"Song {i}") for i in range(4)]
    # Pass 1 (original order): fits 1, 1, 0, 0.5. Pass 2 is shuffled; we recover its order
    # from the judge's own RNG and answer consistently, so averaged labels must match pass 1.
    judge = PlaylistJudge(llm=fake_judge_llm([]), passes=2, seed=3)
    import random
    order = list(range(4))
    random.Random("3:quiet folk").shuffle(order)
    truth = [1, 1, 0, 0.5]
    p1 = {"tracks": [{"n": i + 1, "fit": truth[i]} for i in range(4)], "intent_alignment": 1, "cohesion": 0.5,
          "diversity": 1, "personalization": 1, "appropriateness": 0, "summary": "one off-brief track"}
    p2 = {**p1, "tracks": [{"n": i + 1, "fit": truth[order[i]]} for i in range(4)], "cohesion": 1}
    judge = PlaylistJudge(llm=fake_judge_llm([p1, p2]), passes=2, seed=3)

    out = judge.judge("quiet folk", tracks, ["A"])

    assert out["track_fits"] == [1, 1, 0, 0.5]
    assert out["self_agreement"] == 1.0
    assert out["precision"] == pytest.approx(0.625)
    assert out["dims"]["cohesion"] == 0.75 and out["dims"]["appropriateness"] == 0
    assert out["failed_passes"] == 0


def test_judge_tolerates_bad_passes_and_labels():
    tracks = [t(1), t(2)]
    good = {"tracks": [{"n": 1, "fit": "0.9"}, {"n": 7, "fit": 1}, {"n": "x"}], "intent_alignment": 0.4}
    # Pass 1 (original order) parses; pass 2 (shuffled) returns garbage and is dropped.
    judge = PlaylistJudge(llm=FakeListChatModel(responses=[json.dumps(good), "not json"]), passes=2)
    out = judge.judge("x", tracks, [])
    assert out["failed_passes"] == 1
    assert out["track_fits"] == [1.0, None]
    assert out["dims"]["intent_alignment"] == 0.5
    assert out["self_agreement"] is None


def test_snap():
    assert [snap(v) for v in (0, 0.2, 0.3, 0.74, 0.76, 2, "x")] == [0, 0, 0.5, 0.5, 1, 1, None]


# -- prompts / runner / report ----------------------------------------------------------

def test_gold_prompt_set_is_well_formed():
    prompts = load_prompts()
    assert len(prompts) == 60
    cats = {p["category"] for p in prompts}
    assert cats == {"mood", "activity", "genre_era", "arc", "nuanced", "niche", "personalized"}
    for p in prompts:
        assert p["length"] == 15 and p["profile"] in {"default", "hiphop_rnb"}
        for key in p.get("targets", {}):
            assert key in {"energy", "valence", "tempo", "danceability", "acousticness", "instrumentalness"}
        if "arc" in p:
            metrics.arc_curve(p["arc"], 3)


def test_select_prompts_stratifies():
    picked = select_prompts(load_prompts(), limit=7)
    assert len({p["category"] for p in picked}) == 7


def fixed_intent(text, length):
    return IntentNormalized(mode="curate", description=text,
                            constraints={"energy_range": [0.0, 0.5], "valence_range": [0.0, 1.0],
                                         "bpm_range": [60, 180], "danceability_min": None},
                            structure={"phases": PHASES})


class StubJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, request, tracks, taste):
        self.calls += 1
        fits = [1.0 if "Folk" in x.genres else 0.0 for x in tracks]
        return {"track_fits": fits, "dims": {"intent_alignment": 1.0}, "precision": sum(fits) / max(1, len(fits)),
                "self_agreement": 1.0, "summaries": ["ok"], "failed_passes": 0}


@pytest.fixture
def no_llm():
    with patch("app.agent.llm.provider_available", return_value=False):
        yield


def test_runner_end_to_end_resumes_and_reports(tmp_path, no_llm):
    prompts = [
        {"id": "p1", "category": "mood", "prompt": "quiet folk", "length": 8, "profile": "default",
         "genres_any": ["folk"], "targets": {"energy": [0.0, 0.5]}},
        {"id": "p2", "category": "arc", "prompt": "jazz that builds", "length": 8, "profile": "default",
         "arc": "build"},
    ]
    judge = StubJudge()
    runner = EvalRunner(lambda profile: OfflineCatalog(str(FIXTURE), profile={"seed_artists": ["Willow Creek"]}),
                        judge=judge, parse_intent=fixed_intent)
    configs = [CONFIGS["baseline"], CONFIGS["llm_rank"], CONFIGS["agent"], CONFIGS["agent_seq"]]
    results = runner.run(prompts, configs, "t", out_dir=tmp_path)

    assert len(results["records"]) == 8 and judge.calls == 6  # agent_seq is derived: no extra judging
    assert set(results["intents"]) == {"p1", "p2"}
    rec = next(r for r in results["records"] if r["prompt_id"] == "p1" and r["config"] == "baseline")
    assert rec["metrics"]["genre_hit_rate"] > 0.5 and len(rec["tracks"]) == 8
    assert rec["error"] is None and rec["retrieval_used"] == "heuristic"
    src = next(r for r in results["records"] if r["prompt_id"] == "p2" and r["config"] == "agent")
    seq = next(r for r in results["records"] if r["prompt_id"] == "p2" and r["config"] == "agent_seq")
    assert src["retrieval_used"] == "heuristic"  # agent needs an LLM; falls back cleanly
    assert sorted(x["id"] for x in seq["tracks"]) == sorted(x["id"] for x in src["tracks"])
    assert seq["derived_from"] == "agent"
    # Judge labels follow their tracks through the reorder.
    fit_by_id = dict(zip((x["id"] for x in src["tracks"]), src["judge"]["track_fits"]))
    assert seq["judge"]["track_fits"] == [fit_by_id[x["id"]] for x in seq["tracks"]]

    again = runner.run(prompts, configs, "t", out_dir=tmp_path)  # resume: nothing new to do
    assert judge.calls == 6 and len(again["records"]) == 8

    report = build_report(again)
    assert "## Results" in report and "| Gold score ↑ |" in report
    assert "Paired difference vs `baseline`" in report
    assert "| mood | 1 |" in report


def test_derived_config_must_follow_its_source(tmp_path):
    runner = EvalRunner(lambda p: None, parse_intent=fixed_intent)
    with pytest.raises(ValueError, match="must run before"):
        runner.run([], [CONFIGS["agent_seq"], CONFIGS["agent"]], "x", out_dir=tmp_path)


def test_paired_diff():
    d = paired_diff({"a": 0.2, "b": 0.4, "c": 0.5}, {"a": 0.6, "b": 0.4, "c": 0.9})
    assert d["mean"] == pytest.approx((0.4 + 0 + 0.4) / 3)
    assert d["win_rate"] == pytest.approx(2 / 3) and d["lo"] <= d["mean"] <= d["hi"]
    assert paired_diff({"a": None}, {"a": 1}) is None


def test_label_sheet_and_agreement_roundtrip(tmp_path):
    results = {"run": "r", "records": [
        {"prompt_id": "p1", "config": "baseline", "prompt": "folk",
         "tracks": [t(1).model_dump(), t(2).model_dump()], "judge": {"track_fits": [1.0, 0.25]}},
        {"prompt_id": "p1", "config": "agent", "prompt": "folk",
         "tracks": [t(3).model_dump()], "judge": {"track_fits": [0.0]}},
    ]}
    sheet = tmp_path / "labels.csv"
    assert agreement.write_label_sheet(results, sheet, n=10) == 3
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert "judge" not in rows[0] and all(r["label"] == "" for r in rows)  # blind

    human = {("baseline", "0"): "1", ("baseline", "1"): "0", ("agent", "0"): "0"}
    for r in rows:
        r["label"] = human[(r["config"], r["position"])]
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agreement.LABEL_FIELDS)
        w.writeheader()
        w.writerows(rows)

    pairs = agreement.pair_labels(results, agreement.load_labels(sheet))
    assert sorted(pairs) == [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]  # judge 0.25 snaps to 0
    report = agreement.agreement_report(pairs)
    assert report["n"] == 3 and report["exact_agreement"] == 1.0 and report["weighted_kappa"] == pytest.approx(1.0)


def test_parse_arc_on_unseen_phrasings():
    from app.agent.sequencer import parse_arc

    # Phrasings not in prompts.yaml (the rules were written with those prompts in view).
    assert parse_arc("start slow and get faster as the night goes on") == "build"
    assert parse_arc("something that ramps up for my commute") == "build"
    assert parse_arc("rise to a peak then come back down") == "peak"
    assert parse_arc("help me wind down before bed") == "cooldown"
    assert parse_arc("chill indie for studying") is None
    # Known misses, kept visible rather than tuned away: no arc keyword, so no arc.
    assert parse_arc("from hype to calm") is None
    assert parse_arc("crescendo playlist") is None


def test_parse_intent_sets_arc_phases():
    from app.agent.intent import parse_intent
    from app.agent.sequencer import ARC_PHASES

    with patch("app.agent.intent._call_ollama", side_effect=RuntimeError("no llm")):
        intent = parse_intent("a dinner party that starts mellow and builds into dancing", 12)
        plain = parse_intent("chill indie", 12)
    assert intent.structure["arc"] == "build" and intent.structure["phases"] == ARC_PHASES["build"]
    assert "arc" not in plain.structure
