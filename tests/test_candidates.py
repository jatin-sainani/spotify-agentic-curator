"""Tests for candidate retrieval over a CatalogProvider."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.agent.candidates import build_candidates, build_search_queries
from app.agent.intent import IntentNormalized
from app.catalog.offline import OfflineCatalog
from app.core.draft_models import TrackRef

FIXTURE = Path(__file__).parent / "fixtures" / "catalog_synthetic.csv"
PROFILE = {"name": "test", "seed_artists": ["Gravel Kings"]}

OPEN_CONSTRAINTS = {
    "energy_range": [0.0, 1.0], "valence_range": [0.0, 1.0],
    "bpm_range": [60.0, 180.0], "danceability_min": None,
}


def make_intent(description="calm folk", constraints=None):
    return IntentNormalized(
        mode="curate",
        description=description,
        constraints=constraints or OPEN_CONSTRAINTS,
        structure={"phases": []},
    )


@pytest.fixture(scope="module")
def catalog():
    return OfflineCatalog(str(FIXTURE), profile=PROFILE)


def test_taste_only_without_intent(catalog):
    pool = build_candidates(catalog, limit=10)
    assert len(pool) == 6  # Gravel Kings has 6 tracks
    assert all(t.artists == ["Gravel Kings"] for t in pool)


def test_mixes_taste_and_intent(catalog):
    pool = build_candidates(catalog, limit=20, intent=make_intent(), intent_text="quiet folk for a rainy day")
    artists = {t.artists[0] for t in pool}
    assert "Gravel Kings" in artists  # taste share
    assert {"Willow Creek", "Paper Lanterns"} & artists  # intent search
    assert len(pool) <= 20
    assert len({t.id for t in pool}) == len(pool)


def test_feature_filters_applied_then_relaxed(catalog):
    low_energy = dict(OPEN_CONSTRAINTS, energy_range=[0.0, 0.3])
    pool = build_candidates(
        catalog, limit=40, intent=make_intent("jazz", low_energy), intent_text="jazz"
    )
    jazz = [t for t in pool if "Jazz" in t.genres]
    # Filtered pass returns low-energy jazz first; the relaxed pass tops up the rest.
    assert jazz
    assert jazz[0].audio_features["energy"] <= 0.3


def test_limit_respected(catalog):
    pool = build_candidates(catalog, limit=5, intent=make_intent(), intent_text="folk")
    assert len(pool) == 5


def test_search_failures_are_tolerated():
    cat = Mock()
    cat.name = "mock"
    cat.get_user_taste.side_effect = RuntimeError("boom")
    cat.search.side_effect = RuntimeError("boom")
    assert build_candidates(cat, limit=10, intent=make_intent(), intent_text="folk") == []


def test_works_with_any_provider():
    ref = TrackRef(uri="spotify:track:x", id="x", name="X", artists=["A"])
    cat = Mock()
    cat.name = "mock"
    cat.get_user_taste.return_value = []
    cat.search.return_value = [ref]
    pool = build_candidates(cat, limit=10, intent=make_intent(), intent_text="folk")
    assert pool == [ref]


def test_search_queries_use_genre_hints_and_raw_text():
    queries = build_search_queries(make_intent("Mellow acoustic evening"), "lofi hip hop to study")
    assert queries[0] in {"hip-hop", "acoustic", "lo-fi"}
    assert "lofi hip hop to study" in queries
    assert "Mellow acoustic evening" in queries
    assert len(queries) <= 4


def test_search_queries_never_empty():
    assert build_search_queries(make_intent(""), None) == ["music"]


def test_taste_share_zero_skips_taste(catalog):
    pool = build_candidates(catalog, limit=10, intent=make_intent(), intent_text="jazz", taste_share=0.0)
    assert pool and all(t.artists != ["Gravel Kings"] for t in pool)


def test_genre_hints_prefer_compound_genres():
    from app.agent.candidates import _genre_hints

    assert _genre_hints("80s synth-pop and new wave classics") == ["synthpop", "new wave"]
    assert _genre_hints("2000s emo and pop punk") == ["emo", "pop punk"]
    assert _genre_hints("lo-fi hip hop beats") == ["lo-fi", "hip hop"]
    assert _genre_hints("popcorn and hippos") == []


def test_relaxed_pass_keeps_era():
    calls = []
    cat = Mock()
    cat.name = "mock"
    cat.get_user_taste.return_value = []
    cat.search.side_effect = lambda q, limit, filters: calls.append(dict(filters or {})) or []
    constraints = dict(OPEN_CONSTRAINTS, energy_range=[0.0, 0.3], year_range=[1980, 1989])
    build_candidates(cat, limit=10, intent=make_intent("80s", constraints), intent_text="80s")
    assert any("energy_range" in c for c in calls)                     # strict pass
    assert {"year_range": [1980, 1989]} in calls                         # relaxed pass keeps the era
    assert all("year_range" in c for c in calls)
