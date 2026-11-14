"""Tests for draft editing, export, Spotify confirm and the improve loop."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.agent import draft_ops
from app.agent.draft_ops import DraftError
from app.agent.improve import improve_draft, propose_swaps
from app.catalog.offline import OfflineCatalog
from app.core.draft_models import Draft, TrackRef

FIXTURE = Path(__file__).parent / "fixtures" / "catalog_synthetic.csv"


def ref(tid, artist="A", name=None):
    return TrackRef(uri=f"spotify:track:{tid}", id=tid, name=name or f"T{tid}", artists=[artist])


def make_draft(ids=("a", "b", "c")):
    return Draft(id="d1", intent_text="quiet folk", mode="curate", length=len(ids), tracks=[ref(i) for i in ids])


@pytest.fixture(scope="module")
def catalog():
    return OfflineCatalog(str(FIXTURE), profile={"seed_artists": []})


# -- add / remove / move ----------------------------------------------------

def test_add_track_positions_and_events():
    d = make_draft()
    draft_ops.add_track(d, ref("x"), position=1)
    draft_ops.add_track(d, ref("y"))
    assert [t.id for t in d.tracks] == ["x", "a", "b", "c", "y"]
    assert d.tracks[0].why == "Added by you"
    assert [e["type"] for e in d.feedback_events] == ["add", "add"]
    assert d.feedback_events[0]["position"] == 1


def test_add_rejects_duplicates_and_bad_positions():
    d = make_draft()
    with pytest.raises(DraftError, match="already"):
        draft_ops.add_track(d, ref("a"))
    with pytest.raises(DraftError, match="between 1 and 4"):
        draft_ops.add_track(d, ref("z"), position=9)


def test_remove_track():
    d = make_draft()
    removed = draft_ops.remove_track(d, "b")
    assert removed.id == "b" and [t.id for t in d.tracks] == ["a", "c"]
    assert d.feedback_events[-1] == {**d.feedback_events[-1], "type": "remove", "track_id": "b"}
    with pytest.raises(DraftError, match="not in draft"):
        draft_ops.remove_track(d, "b")


def test_move_track():
    d = make_draft()
    draft_ops.move_track(d, "c", 1)
    assert [t.id for t in d.tracks] == ["c", "a", "b"]
    assert d.feedback_events[-1]["from_position"] == 3 and d.feedback_events[-1]["to_position"] == 1
    with pytest.raises(DraftError):
        draft_ops.move_track(d, "c", 0)


# -- rerank -----------------------------------------------------------------

def test_rerank_by_features_without_llm(catalog):
    tracks = catalog.get_tracks(["synth0013", "synth0001", "synth0049"])  # electronic, folk, jazz
    d = Draft(id="d2", intent_text="mixed", mode="curate", length=3, tracks=tracks)
    low_energy = {"energy_range": [0.0, 0.3], "valence_range": [0.0, 1.0], "bpm_range": [60, 180], "danceability_min": None}
    from app.agent.intent import IntentNormalized
    with patch("app.agent.intent.parse_intent",
               return_value=IntentNormalized(mode="curate", description="calm", constraints=low_energy, structure={})):
        draft_ops.rerank(d, "calmest first", use_llm=False)
    assert d.tracks[-1].genres[0] == "Electronic"
    assert {t.id for t in d.tracks} == {"synth0013", "synth0001", "synth0049"}
    assert d.feedback_events[-1]["type"] == "rerank" and d.feedback_events[-1]["llm"] is False


def test_rerank_empty_draft():
    with pytest.raises(DraftError):
        draft_ops.rerank(make_draft(()), "x", use_llm=False)


# -- export / confirm ---------------------------------------------------------

def test_export_uris_and_json(tmp_path):
    d = make_draft()
    uris = draft_ops.export_draft(d, "uris", tmp_path)
    assert uris.read_text().split() == ["spotify:track:a", "spotify:track:b", "spotify:track:c"]
    data = json.loads(draft_ops.export_draft(d, "json", tmp_path).read_text())
    assert data["request"] == "quiet folk" and len(data["tracks"]) == 3
    with pytest.raises(DraftError):
        draft_ops.export_draft(d, "m3u", tmp_path)


def test_confirm_to_spotify_marks_draft():
    client = MagicMock()
    client.create_playlist.return_value = {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}
    d = make_draft()
    draft_ops.confirm_to_spotify(d, client, "My mix", "desc")
    client.add_items_to_playlist.assert_called_once_with("pl1", ["spotify:track:a", "spotify:track:b", "spotify:track:c"])
    assert d.status == "confirmed" and d.external_url.endswith("pl1")
    assert d.feedback_events[-1]["type"] == "confirm"


def test_spotify_client_playlist_endpoints():
    from app.spotify.client import SpotifyClient

    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path, json.loads(request.content or b"null")))
        if request.url.path.endswith("/me/playlists"):
            return httpx.Response(201, json={"id": "pl9"})
        return httpx.Response(201, json={"snapshot_id": "s"})

    auth = MagicMock()
    auth.get_valid_access_token.return_value = "tok"
    client = SpotifyClient.__new__(SpotifyClient)
    client.auth = auth
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client._get_auth_headers = lambda: {"Authorization": "Bearer tok"}

    assert client.create_playlist("Mix", "d")["id"] == "pl9"
    client.add_items_to_playlist("pl9", [f"spotify:track:{i}" for i in range(150)])

    assert calls[0][:2] == ("POST", "/v1/me/playlists") and calls[0][2]["public"] is False
    assert [c[1] for c in calls[1:]] == ["/v1/playlists/pl9/items"] * 2
    assert [len(c[2]["uris"]) for c in calls[1:]] == [100, 50]


# -- improve loop -------------------------------------------------------------

@pytest.fixture
def no_llm():
    with patch("app.agent.intent._call_ollama", side_effect=RuntimeError("no llm")), \
         patch("app.agent.llm.provider_available", return_value=False):
        yield


def calm_intent():
    from app.agent.intent import IntentNormalized

    return IntentNormalized(mode="curate", description="calm folk",
                            constraints={"energy_range": [0.0, 0.4], "valence_range": [0.0, 1.0],
                                         "bpm_range": [60, 180], "danceability_min": None},
                            structure={})


def loud_draft(catalog):
    # Two loud electronic tracks + one folk track; feedback asks for calm folk.
    tracks = catalog.get_tracks(["synth0013", "synth0014", "synth0001"])
    return Draft(id="imp", intent_text="sunday playlist", mode="curate", length=3, tracks=tracks)


def test_propose_swaps_targets_weakest_tracks(catalog, no_llm):
    with patch("app.agent.intent.parse_intent", return_value=calm_intent()):
        proposals = propose_swaps(loud_draft(catalog), "calmer, more folk", catalog, max_swaps=2, use_llm=False)
    assert len(proposals) == 2
    assert {p["remove"]["genres"][0] for p in proposals} == {"Electronic"}
    assert all(p["new_score"] > p["old_score"] for p in proposals)
    assert all(p["add"]["audio_features"]["energy"] <= 0.4 for p in proposals)


def test_improve_human_in_the_loop(catalog, no_llm):
    seen = {}

    def decide(proposals):
        seen["n"] = len(proposals)
        return [True, False]  # accept first, reject second

    draft = loud_draft(catalog)
    with patch("app.agent.intent.parse_intent", return_value=calm_intent()):
        result = improve_draft(draft, "calmer, more folk", catalog, decide, max_swaps=2, use_llm=False)

    assert seen["n"] == 2 and result["applied"] == 1
    new = result["draft"]
    assert len(new.tracks) == 3
    assert sum(1 for t in new.tracks if t.genres[0] == "Electronic") == 1
    kinds = [e["type"] for e in new.feedback_events]
    assert kinds == ["swap_accepted", "swap_rejected"]
    assert new.tracks[[t.id for t in draft.tracks].index(result["proposals"][0]["remove"]["id"])].why.startswith("Swapped in")


def test_improve_without_proposals_skips_review(catalog, no_llm):
    decide = MagicMock()
    with patch("app.agent.improve.propose_swaps", return_value=[]):
        result = improve_draft(loud_draft(catalog), "x", catalog, decide)
    decide.assert_not_called()
    assert result["applied"] == 0


def test_improve_rejects_wrong_decision_count(catalog, no_llm):
    with patch("app.agent.intent.parse_intent", return_value=calm_intent()), pytest.raises(ValueError, match="one boolean"):
        improve_draft(loud_draft(catalog), "calmer", catalog, lambda p: [True], max_swaps=2, use_llm=False)


def test_propose_swaps_requires_good_replacements(catalog, no_llm):
    # Replacements that only beat a bad track but are weak themselves are not proposed.
    with patch("app.agent.intent.parse_intent", return_value=calm_intent()), \
         patch("app.agent.improve._score_all", side_effect=lambda tracks, *a: {
             t.id: (0.1 if t.id in {"synth0013", "synth0014", "synth0001"} else 0.3) for t in tracks}):
        assert propose_swaps(loud_draft(catalog), "calmer", catalog, use_llm=False) == []


def test_propose_swaps_never_offers_a_song_already_in_draft(catalog, no_llm):
    draft = loud_draft(catalog)
    dup = draft.tracks[2].model_copy(update={"id": "dupe", "uri": "spotify:track:dupe"})
    with patch("app.agent.intent.parse_intent", return_value=calm_intent()), \
         patch("app.agent.candidates.build_candidates", return_value=[dup]):
        assert propose_swaps(draft, "calmer", catalog, use_llm=False) == []
