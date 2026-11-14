"""Tests for the offline catalog provider (synthetic fixture, no network)."""

from pathlib import Path

import pandas as pd
import pytest

from app.catalog.base import CatalogProvider, FeatureFilters
from app.catalog.offline import OfflineCatalog, _parse_list, build_index, tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "catalog_synthetic.csv"
PROFILE = {"name": "test", "seed_artists": ["Willow Creek", "Neon Atlas"], "seed_track_ids": ["synth0013"]}


@pytest.fixture(scope="module")
def catalog():
    return OfflineCatalog(str(FIXTURE), profile=PROFILE)


def test_implements_protocol(catalog):
    assert isinstance(catalog, CatalogProvider)
    assert catalog.name == "offline"
    assert len(catalog) == 60


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="catalog build"):
        OfflineCatalog(str(tmp_path / "nope.parquet"))


def test_parse_list_cells():
    assert _parse_list('["a", "b"]') == ["a", "b"]
    assert _parse_list("[]") == []
    assert _parse_list(None) == []
    assert _parse_list("plain") == ["plain"]


def test_tokenize_drops_stopwords():
    assert tokenize("Some rainy folk songs for my Sunday") == ["rainy", "folk", "sunday"]


def test_search_ranks_genre_matches_first(catalog):
    results = catalog.search("folk", limit=5)
    assert len(results) == 5
    assert all("Folk" in r.genres for r in results)


def test_search_attaches_features_and_metadata(catalog):
    ref = catalog.search("house", limit=1)[0]
    assert ref.uri == f"spotify:track:{ref.id}"
    assert {"energy", "valence", "tempo", "danceability"} <= set(ref.audio_features)
    assert ref.year is not None and ref.popularity is not None


def test_search_applies_feature_filters(catalog):
    filters = FeatureFilters(energy_range=[0.0, 0.5])
    results = catalog.search("song", limit=60, filters=filters)
    assert results
    assert all(r.audio_features["energy"] <= 0.5 for r in results)


def test_search_matches_lyrics_words(catalog):
    results = catalog.search("midnight", limit=10)
    assert results and all("Jazz" in r.genres for r in results)


def test_search_no_match_returns_empty(catalog):
    assert catalog.search("zzzzqqq", limit=5) == []


def test_get_tracks_and_features_skip_unknown_ids(catalog):
    refs = catalog.get_tracks(["synth0001", "missing"])
    assert [r.id for r in refs] == ["synth0001"]
    feats = catalog.features(["synth0001", "missing"])
    assert set(feats) == {"synth0001"}


def test_user_taste_uses_seed_tracks_then_round_robins_artists(catalog):
    taste = catalog.get_user_taste(limit=6)
    assert taste[0].id == "synth0013"
    first_artists = [t.artists[0] for t in taste[1:3]]
    assert set(first_artists) == {"Willow Creek", "Neon Atlas"}
    assert len({t.id for t in taste}) == len(taste) == 6


def test_top_artists(catalog):
    assert catalog.top_artists() == {"Willow Creek", "Neon Atlas"}


def test_named_profile_missing_falls_back_to_empty():
    cat = OfflineCatalog(str(FIXTURE), profile="does-not-exist")
    assert cat.top_artists() == set()
    assert cat.get_user_taste(limit=5) == []


def test_feature_filters_from_constraints():
    f = FeatureFilters.from_constraints({
        "energy_range": [0.2, 0.6], "valence_range": [0.0, 1.0],
        "bpm_range": [60, 180], "danceability_min": None,
    })
    assert f["energy_range"] == [0.2, 0.6]
    assert "danceability_min" not in f
    assert not f.is_unconstrained()
    assert FeatureFilters.from_constraints({
        "energy_range": [0.0, 1.0], "valence_range": [0.0, 1.0], "bpm_range": [60, 180],
    }).is_unconstrained()


def test_build_index_sample_and_csv_output(tmp_path):
    raw = pd.read_csv(FIXTURE).rename(columns={"lyrics_snippet": "lyrics"})
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    out = tmp_path / "idx.csv"

    df = build_index(str(raw_path), str(out), sample=20)

    assert out.exists()
    assert len(df) == 20
    assert set(df["genre"]) == {"Folk", "Electronic", "Rock", "Hip-Hop", "Jazz"}
    assert "lyrics" not in df.columns and "lyrics_snippet" in df.columns
    assert len(OfflineCatalog(str(out), profile=PROFILE)) == 20


def test_build_index_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,name\n1,x\n")
    with pytest.raises(ValueError, match="missing required columns"):
        build_index(str(bad), str(tmp_path / "o.csv"))


# -- regressions found by the first eval run (genre-01: "80s synth-pop and new wave") --

def test_parse_era():
    from app.catalog.base import parse_era

    assert parse_era("80s synth-pop") == (1980, 1989)
    assert parse_era("classic 1960s soul") == (1960, 1969)
    assert parse_era("'90s grunge") == (1990, 1999)
    assert parse_era("2000s emo and pop punk") == (2000, 2009)
    assert parse_era("70s and 80s rock") == (1970, 1989)
    assert parse_era("the nineties") == (1990, 1999)
    assert parse_era("songs for 20 minutes of running") is None
    assert parse_era("") is None


def test_search_matches_whole_words_only():
    df = pd.DataFrame([
        {"id": "h", "name": "I Want a Hippopotamus", "album_name": "x", "artists": '["Kid"]', "genre": "Pop",
         "niche_genres": '["christmas"]', "year": 1953, "popularity": 90, "energy": 0.5, "valence": 0.5,
         "tempo": 100, "danceability": 0.5},
        {"id": "s", "name": "Tainted Love", "album_name": "x", "artists": '["Soft Cell"]', "genre": "Electronic",
         "niche_genres": '["synthpop", "new wave"]', "year": 1981, "popularity": 10, "energy": 0.5,
         "valence": 0.5, "tempo": 100, "danceability": 0.5},
    ])
    cat = OfflineCatalog("unused", profile=PROFILE, df=df)
    assert [r.id for r in cat.search("hippo")] == []           # no substring hits...
    assert [r.id for r in cat.search("synth-pop")] == ["s"]     # ...hyphen variants match "synthpop"
    assert [r.id for r in cat.search("new wave")] == ["s"]
    assert [r.id for r in cat.search("synthpop", filters=FeatureFilters(year_range=[1980, 1989]))] == ["s"]
    assert cat.search("synthpop", filters=FeatureFilters(year_range=[1990, 1999])) == []


def test_phrase_match_outranks_separate_words():
    df = pd.DataFrame([
        {"id": "w", "name": "New Song", "album_name": "Wave", "artists": '["A"]', "genre": "Pop",
         "niche_genres": '["pop"]', "year": 2020, "popularity": 99, "energy": 0.5, "valence": 0.5,
         "tempo": 100, "danceability": 0.5},
        {"id": "nw", "name": "Track", "album_name": "x", "artists": '["B"]', "genre": "Rock",
         "niche_genres": '["new wave"]', "year": 1982, "popularity": 1, "energy": 0.5, "valence": 0.5,
         "tempo": 100, "danceability": 0.5},
    ])
    cat = OfflineCatalog("unused", profile=PROFILE, df=df)
    assert cat.search("new wave", limit=2)[0].id == "nw"


def test_feature_filters_carry_era():
    f = FeatureFilters.from_constraints({"energy_range": [0.2, 0.6], "year_range": [1980, 1989]})
    assert f["year_range"] == [1980, 1989]
    assert dict(f.era_only()) == {"year_range": [1980, 1989]}
    assert not FeatureFilters(year_range=[1980, 1989]).is_unconstrained()
