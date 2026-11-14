"""Tests for track ranking and selection (LLM always mocked)."""

from collections import Counter
from unittest.mock import patch

import pytest

from app.agent.intent import IntentNormalized
from app.agent.ranker import (
    RankWeights,
    _calculate_constraint_fit,
    _calculate_diversity_penalty,
    _generate_selection_reason,
    _stage1_pre_score,
    format_track_line,
    rank_and_select,
    select_with_diversity,
)
from app.core.draft_models import TrackRef


def make_intent(**constraints):
    base = {"energy_range": [0.0, 1.0], "valence_range": [0.0, 1.0], "bpm_range": [60, 200], "danceability_min": None}
    base.update(constraints)
    return IntentNormalized(mode="curate", description="Test playlist", constraints=base, structure={"phases": []})


def track(tid, artist="A", **features):
    return TrackRef(uri=f"spotify:track:{tid}", id=str(tid), name=f"Track {tid}", artists=[artist],
                    audio_features=features or None)


@pytest.fixture
def no_llm_calls():
    """Fail loudly if anything tries to reach a real LLM."""
    with patch("app.agent.llm.invoke_score_chain", side_effect=AssertionError("LLM called")):
        yield


class TestStage1:
    def test_basic_ranking_without_llm(self, no_llm_calls):
        intent = make_intent(energy_range=[0.5, 0.8], valence_range=[0.3, 0.7], bpm_range=[100, 140], danceability_min=0.4)
        candidates = [
            track(1, "Artist A", energy=0.9, tempo=120, valence=0.5, danceability=0.6),
            track(2, "Artist B", energy=0.3, tempo=120, valence=0.5, danceability=0.6),
            track(3, "Artist C", energy=0.65, tempo=120, valence=0.5, danceability=0.7),
        ]
        selected = rank_and_select(candidates, intent, 2, {"Artist C"}, use_llm=False)

        assert [t.id for t in selected] == ["3", "1"]  # perfect fit, then the closer miss
        assert "excellent overall match" in selected[0].why
        assert "from your top artists" in selected[0].why
        assert "energy level fits" in selected[0].why

    def test_out_of_range_tracks_fill_remaining_slots(self, no_llm_calls):
        intent = make_intent(energy_range=[0.9, 0.95], valence_range=[0.9, 0.95], bpm_range=[170, 175], danceability_min=0.95)
        candidates = [
            track(2, "B", energy=0.5, tempo=100, valence=0.5, danceability=0.5),
            track(1, "A", energy=0.92, tempo=172, valence=0.92, danceability=0.96),
        ]
        selected = rank_and_select(candidates, intent, 2, set(), use_llm=False)
        assert [t.id for t in selected] == ["1", "2"]

    def test_unknown_features_are_neutral(self):
        intent = make_intent(energy_range=[0.0, 0.2])
        scored = dict((t.id, s) for t, s in _stage1_pre_score(
            [track("known_bad", energy=0.9), TrackRef(uri="spotify:track:u", id="u", name="U", artists=["X"])],
            intent, set()))
        assert scored["u"] > scored["known_bad"]

    def test_weights_are_configurable(self):
        intent = make_intent(energy_range=[0.0, 0.2])
        fav = track("fav", "Fav", energy=0.9)
        fit = track("fit", "Other", energy=0.1)
        taste_heavy = [t.id for t, _ in _stage1_pre_score([fav, fit], intent, {"Fav"}, RankWeights(fit=0.1, taste=0.8, base=0.1))]
        fit_heavy = [t.id for t, _ in _stage1_pre_score([fav, fit], intent, {"Fav"}, RankWeights(fit=0.8, taste=0.1, base=0.1))]
        assert taste_heavy == ["fav", "fit"]
        assert fit_heavy == ["fit", "fav"]


class TestConstraintFit:
    def test_inside_range_is_perfect_regardless_of_position(self):
        intent_c = {"energy_range": [0.2, 0.8]}
        at_edge = _calculate_constraint_fit(track(1, energy=0.2), intent_c)
        centered = _calculate_constraint_fit(track(2, energy=0.5), intent_c)
        assert at_edge == centered == 1.0

    def test_decays_outside_range(self):
        c = {"energy_range": [0.6, 0.8], "valence_range": [0.4, 0.6], "bpm_range": [110, 130], "danceability_min": 0.5}
        good = _calculate_constraint_fit(track(1, energy=0.7, tempo=120, valence=0.5, danceability=0.6), c)
        near = _calculate_constraint_fit(track(2, energy=0.9, tempo=120, valence=0.5, danceability=0.6), c)
        poor = _calculate_constraint_fit(track(3, energy=0.7, tempo=120, valence=0.5, danceability=0.6),
                                         {"energy_range": [0.1, 0.2], "valence_range": [0.8, 0.9],
                                          "bpm_range": [200, 210], "danceability_min": 0.9})
        assert good == 1.0
        assert 0.6 < near < 1.0
        assert poor < 0.3

    def test_no_features(self):
        assert _calculate_constraint_fit(TrackRef(uri="spotify:track:x", id="x", name="X", artists=[]), {"energy_range": [0, 1]}) == 0.0


class TestDiversity:
    def test_penalty_steps(self):
        t = track(1, "Artist A")
        assert _calculate_diversity_penalty(t, Counter()) == 1.0
        assert _calculate_diversity_penalty(t, Counter(["Artist A"])) == 0.8
        assert _calculate_diversity_penalty(t, Counter(["Artist A"] * 2)) == 0.5
        assert _calculate_diversity_penalty(t, Counter(["Artist A"] * 3)) == 0.2

    def test_artist_cap_without_shortfall(self, no_llm_calls):
        candidates = [track(i, "Popular", energy=0.7, tempo=120, valence=0.5, danceability=0.8) for i in range(5)]
        candidates.append(track("other", "Other", energy=0.7, tempo=120, valence=0.5, danceability=0.8))
        selected = rank_and_select(candidates, make_intent(), 4, {"Popular"}, use_llm=False)

        assert len(selected) == 4
        assert Counter(t.artists[0] for t in selected) == {"Popular": 3, "Other": 1}

    def test_artist_cap_is_soft_when_pool_is_thin(self):
        items = [(track(i, "Solo"), 1.0 - i * 0.1, {}) for i in range(5)] + [(track("o", "Other"), 0.05, {})]
        picked = select_with_diversity(items, 5)
        assert len(picked) == 5
        # Cap admits 0,1,2 then "o"; the best skipped track (3) fills the last slot.
        assert [t.id for t, _, _ in picked] == ["0", "1", "2", "3", "o"]

    def test_select_with_diversity_skips_duplicates(self):
        t = track(1)
        picked = select_with_diversity([(t, 0.9, {}), (t, 0.8, {})], 5)
        assert len(picked) == 1


class TestStage2:
    def _candidates(self, n):
        return [track(i, f"Artist {i}", energy=0.5, tempo=120, valence=0.5, danceability=0.5) for i in range(n)]

    def test_llm_relevance_reorders_and_batches(self):
        calls = []

        def fake_chain(block, description, priors):
            calls.append(block)
            lines = block.splitlines()
            # Score the last track of each batch highest.
            return {"scores": [{"n": n, "relevance": 95 if n == len(lines) else 10, "mood_tags": ["calm"], "why": "fits"}
                               for n in range(1, len(lines) + 1)]}

        with patch("app.agent.llm.invoke_score_chain", side_effect=fake_chain):
            selected = rank_and_select(self._candidates(12), make_intent(), 2, set(), use_llm=True)

        assert len(calls) == 2  # 12 tracks, batch size 10
        assert {t.id for t in selected} == {"9", "11"}
        assert "highly relevant" in selected[0].why and "mood: calm" in selected[0].why

    def test_partial_llm_response_defaults_missing(self):
        with patch("app.agent.llm.invoke_score_chain", return_value={"scores": [{"n": 1, "relevance": "90"}]}):
            selected = rank_and_select(self._candidates(3), make_intent(), 3, set(), use_llm=True)
        assert selected[0].id == "0"
        assert len(selected) == 3

    def test_failed_batch_falls_back(self):
        with patch("app.agent.llm.invoke_score_chain", side_effect=RuntimeError("timeout")):
            selected = rank_and_select(self._candidates(3), make_intent(), 3, set(), use_llm=True)
        assert len(selected) == 3

    def test_skips_llm_when_provider_unavailable(self, no_llm_calls):
        with patch("app.agent.llm.provider_available", return_value=False):
            selected = rank_and_select(self._candidates(3), make_intent(), 2, set())
        assert len(selected) == 2

    def test_track_line_includes_evidence(self):
        t = track(1, "Artist", energy=0.25, valence=0.4, tempo=92.4)
        t.genres = ["Folk", "indie folk"]
        t.year = 2011
        line = format_track_line(3, t)
        assert line.startswith('3. "Track 1" by Artist')
        assert "genres: indie folk (catalog bucket: Folk)" in line and "year: 2011" in line
        assert "energy=0.25" in line and "tempo=92" in line
        t.genres = ["Jazz"]
        assert "genre: Jazz" in format_track_line(1, t)

    def test_scorer_sees_raw_request_and_feedback(self):
        seen = []

        def fake_chain(block, description, priors):
            seen.append(description)
            return {"scores": [{"n": 1, "relevance": 50}]}

        intent = make_intent()  # description "Test playlist"
        with patch("app.agent.llm.invoke_score_chain", side_effect=fake_chain):
            rank_and_select(self._candidates(1), intent, 1, set(), use_llm=True,
                            request_text="80s synth-pop. Feedback: less mainstream")
        assert seen == ["80s synth-pop. Feedback: less mainstream (interpreted as: Test playlist)"]


class TestSongIdentity:
    def test_song_key_collapses_versions(self):
        from app.agent.ranker import song_key

        base = TrackRef(uri="spotify:track:1", id="1", name="Blue Monday '88", artists=["New Order"])
        v12 = TrackRef(uri="spotify:track:2", id="2", name="Blue Monday ('88) - 12\" Version", artists=["New Order"])
        remaster = TrackRef(uri="spotify:track:3", id="3", name="Blue Monday - 2015 Remaster", artists=["New Order"])
        cover = TrackRef(uri="spotify:track:4", id="4", name="Blue Monday", artists=["Orgy"])
        live = TrackRef(uri="spotify:track:5", id="5", name="Blue Monday [Live]", artists=["New Order"])
        feat = TrackRef(uri="spotify:track:6", id="6", name="I Love It (feat. Charli XCX)", artists=["Icona Pop"])
        plain = TrackRef(uri="spotify:track:7", id="7", name="I Love It", artists=["Icona Pop"])
        assert song_key(base) == song_key(v12)           # "'88" kept, brackets ignored
        assert song_key(remaster) == song_key(live)      # version markers dropped
        assert song_key(feat) == song_key(plain)
        assert song_key(remaster) != song_key(cover)     # same title, different artist

    def test_selection_never_repeats_a_song(self):
        a = TrackRef(uri="spotify:track:a", id="a", name="I Love It (feat. Charli XCX)", artists=["Icona Pop"])
        b = TrackRef(uri="spotify:track:b", id="b", name="I Love It (feat. Charli XCX)", artists=["Icona Pop"])
        c = track("c", "Other")
        picked = select_with_diversity([(a, 0.9, {}), (b, 0.8, {}), (c, 0.1, {})], 3)
        assert [t.id for t, _, _ in picked] == ["a", "c"]


def test_legacy_selection_reason():
    t = track(1, "Test Artist", energy=0.7, tempo=120, valence=0.5, danceability=0.8)
    intent = make_intent(energy_range=[0.6, 0.8], valence_range=[0.4, 0.6], bpm_range=[110, 130], danceability_min=0.5)
    reason = _generate_selection_reason(t, 0.9, intent, {"Test Artist"}, 0)
    for phrase in ("excellent overall match", "energy level fits", "mood matches", "tempo fits",
                   "danceable enough", "from your top artists"):
        assert phrase in reason
    assert "constraint relaxation (round 2)" in _generate_selection_reason(t, 0.7, intent, set(), 2)


def test_empty_candidates():
    assert rank_and_select([], make_intent(), 5, set(), use_llm=False) == []


def test_constraint_fit_includes_era():
    c = {"energy_range": [0.0, 1.0], "year_range": [1980, 1989]}
    t80 = track("a", energy=0.5)
    t80.year = 1984
    t95 = track("b", energy=0.5)
    t95.year = 1995
    t90 = track("c", energy=0.5)
    t90.year = 1991
    assert _calculate_constraint_fit(t80, c) == 1.0
    assert _calculate_constraint_fit(t95, c) == 0.5       # energy fits, era is 6y off (tolerance 4)
    assert 0.5 < _calculate_constraint_fit(t90, c) < 1.0  # 2y off: partial credit


def test_parse_intent_adds_era_from_text():
    from app.agent.intent import parse_intent

    with patch("app.agent.intent._call_ollama", side_effect=RuntimeError("no llm")):
        intent = parse_intent("90s grunge for driving", 10)
    assert intent.constraints["year_range"] == [1990, 1999]


def test_adaptive_taste_weight():
    w = RankWeights(taste_when_impersonal=0.1)
    assert w.stage1_weights(personal=True) == (0.4, 0.4)
    assert w.stage1_weights(personal=False) == pytest.approx((0.7, 0.1))
    assert RankWeights().stage1_weights(personal=False) == (0.4, 0.4)  # default: not adaptive

    fav, fit = track("fav", "Fav", energy=0.9), track("fit", "Other", energy=0.1)
    impersonal = make_intent(energy_range=[0.0, 0.2])
    personal = make_intent(energy_range=[0.0, 0.2]).model_copy(update={"structure": {"personal": True}})
    def order(intent):
        return [t.id for t, _ in _stage1_pre_score([fav, fit], intent, {"Fav"}, w)]
    assert order(impersonal) == ["fit", "fav"]
    assert order(personal) == ["fav", "fit"]


def test_is_personal_request():
    from app.agent.intent import is_personal_request

    assert is_personal_request("songs like my favorites but calmer")
    assert is_personal_request("new artists similar to what I already like")
    assert not is_personal_request("put me in a good mood")
    assert not is_personal_request("80s synth-pop")
