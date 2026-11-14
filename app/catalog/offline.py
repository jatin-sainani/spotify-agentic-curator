"""Offline catalog backed by a local songs dataset.

Build the index once from the raw dataset CSV:

    python -m app catalog build --csv songs.csv            # full index
    python -m app catalog build --csv songs.csv --sample 20000

The index keeps metadata and audio features plus a short lyrics snippet for
search. It is written to data/ (gitignored) and is never committed: it is
derived from Spotify metadata and copyrighted lyrics.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from app.catalog.base import FEATURE_KEYS, FeatureFilters
from app.core.draft_models import TrackRef
from core.logging import logger

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

INDEX_COLUMNS = [
    "id", "name", "album_name", "artists", "genre", "niche_genres", "year",
    "popularity", *FEATURE_KEYS,
]
LYRICS_SNIPPET_CHARS = 300

_TOKEN_RE = re.compile(r"[a-z0-9&'\-]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "with", "my",
    "me", "i", "some", "songs", "song", "music", "tracks", "track", "playlist",
    "that", "this", "is", "are", "be", "like", "vibes", "vibe", "feel", "feeling",
}


def _parse_list(value: Any) -> List[str]:
    """Parse a JSON-ish list cell like '["a", "b"]' into a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (ValueError, TypeError):
        pass
    return [value.strip("[]\" ")] if value.strip("[]\" ") else []


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _word_pattern(token: str) -> str:
    """Regex matching a token as whole word(s); hyphen/space/no-separator variants match."""
    parts = [re.escape(p) for p in re.split(r"[-\s]+", token) if p]
    return r"(?<![a-z0-9])" + r"[-\s]?".join(parts) + r"(?![a-z0-9])"


def build_index(
    csv_path: str,
    out_path: str,
    sample: Optional[int] = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Build the offline catalog index from the raw dataset CSV.

    Args:
        csv_path: Raw dataset with the INDEX_COLUMNS (+ optional lyrics).
        out_path: Output .parquet (needs pyarrow) or .csv/.csv.gz path.
        sample: If set, keep a genre-stratified sample of this many tracks,
            biased toward popular tracks so demos are recognizable.
    """
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [c for c in INDEX_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    usecols = INDEX_COLUMNS + (["lyrics"] if "lyrics" in header else [])

    logger.info(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.dropna(subset=["id", "name"]).drop_duplicates(subset="id")

    if "lyrics" in df.columns:
        df["lyrics_snippet"] = df["lyrics"].fillna("").astype(str).str.slice(0, LYRICS_SNIPPET_CHARS)
        df = df.drop(columns=["lyrics"])
    else:
        df["lyrics_snippet"] = ""

    if sample and sample < len(df):
        # Stratify by genre, then take the most popular within each genre.
        shares = df["genre"].value_counts(normalize=True)
        parts = []
        for genre, share in shares.items():
            n = max(1, int(round(share * sample)))
            pool = df[df["genre"] == genre].nlargest(n * 3, "popularity")
            parts.append(pool.sample(n=min(n, len(pool)), random_state=seed))
        df = pd.concat(parts).head(sample)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} tracks to {out}")
    return df


class OfflineCatalog:
    """CatalogProvider over a local index (parquet or csv)."""

    name = "offline"

    def __init__(
        self,
        index_path: str,
        profile: Union[str, Dict[str, Any]] = "default",
        df: Optional[pd.DataFrame] = None,
    ):
        if df is None:
            path = Path(index_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Offline catalog index not found at '{path}'. Build it with:\n"
                    f"  python -m app catalog build --csv <songs.csv> --out {path}"
                )
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        self._df = self._prepare(df)
        self._by_id = self._df.set_index("id", drop=False)
        self.profile = load_profile(profile) if isinstance(profile, str) else {
            "name": "custom", "seed_artists": [], "seed_track_ids": [], **profile
        }
        logger.info(f"Offline catalog loaded: {len(self._df)} tracks, profile '{self.profile['name']}'")

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["artists_list"] = df["artists"].map(_parse_list)
        df["niche_list"] = df["niche_genres"].map(_parse_list) if "niche_genres" in df else [[]] * len(df)
        if "lyrics_snippet" not in df:
            df["lyrics_snippet"] = ""
        df["lyrics_snippet"] = df["lyrics_snippet"].fillna("").astype(str)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        # Lower-cased search fields, weighted by how strong a signal they are.
        df["_genre_text"] = (
            df["genre"].fillna("").astype(str) + " " + df["niche_list"].map(" ".join)
        ).str.lower()
        df["_meta_text"] = (
            df["name"].astype(str) + " " + df["artists_list"].map(" ".join) + " "
            + df["album_name"].fillna("").astype(str)
        ).str.lower()
        df["_lyrics_text"] = df["lyrics_snippet"].str.lower()
        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    # -- conversion ---------------------------------------------------------

    def _to_ref(self, row: pd.Series) -> TrackRef:
        features = {k: float(row[k]) for k in FEATURE_KEYS if k in row and pd.notna(row[k])}
        return TrackRef(
            uri=f"spotify:track:{row['id']}",
            id=str(row["id"]),
            name=str(row["name"]),
            artists=list(row["artists_list"]),
            audio_features=features or None,
            genres=[str(row["genre"])] + list(row["niche_list"]) if pd.notna(row.get("genre")) else list(row["niche_list"]),
            year=int(row["year"]) if pd.notna(row.get("year")) else None,
            popularity=int(row["popularity"]),
        )

    # -- CatalogProvider ----------------------------------------------------

    def search(self, query: str, limit: int = 20, filters: Optional[FeatureFilters] = None) -> List[TrackRef]:
        df = self._apply_filters(self._df, filters)
        tokens = tokenize(query)
        if tokens:
            score = pd.Series(0.0, index=df.index)
            for tok in tokens:
                # Whole words only ("pop" must not match "hippopotamus"); hyphenated
                # tokens also match their joined/spaced forms ("synth-pop" -> "synthpop").
                pattern = _word_pattern(tok)
                score += 3.0 * df["_genre_text"].str.contains(pattern, regex=True)
                score += 2.0 * df["_meta_text"].str.contains(pattern, regex=True)
                score += 1.0 * df["_lyrics_text"].str.contains(pattern, regex=True)
            # Multi-word genres ("new wave", "trip hop") outrank their separate words.
            for a, b in zip(tokens, tokens[1:]):
                score += 4.0 * df["_genre_text"].str.contains(_word_pattern(f"{a} {b}"), regex=True)
            df = df.assign(_score=score)
            df = df[df["_score"] > 0]
            # Text relevance first, popularity as tie-breaker.
            df = df.sort_values(["_score", "popularity"], ascending=[False, False])
        else:
            df = df.sort_values("popularity", ascending=False)
        return [self._to_ref(row) for _, row in df.head(limit).iterrows()]

    def get_tracks(self, track_ids: List[str]) -> List[TrackRef]:
        found = [tid for tid in track_ids if tid in self._by_id.index]
        return [self._to_ref(self._by_id.loc[tid]) for tid in found]

    def features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return {ref.id: ref.audio_features for ref in self.get_tracks(track_ids) if ref.audio_features}

    def get_user_taste(self, limit: int = 50) -> List[TrackRef]:
        refs = self.get_tracks(self.profile.get("seed_track_ids", []))
        seen = {r.id for r in refs}
        artists = {a.lower() for a in self.profile.get("seed_artists", [])}
        if artists and len(refs) < limit:
            mask = self._df["artists_list"].map(lambda lst: any(a.lower() in artists for a in lst))
            pool = self._df[mask].sort_values("popularity", ascending=False)
            # Round-robin across artists so one prolific artist doesn't dominate.
            pool = pool.assign(_rank=pool.groupby(pool["artists_list"].str[0]).cumcount())
            pool = pool.sort_values(["_rank", "popularity"], ascending=[True, False])
            for _, row in pool.iterrows():
                if len(refs) >= limit:
                    break
                if row["id"] not in seen:
                    refs.append(self._to_ref(row))
                    seen.add(row["id"])
        return refs[:limit]

    def top_artists(self) -> Set[str]:
        return set(self.profile.get("seed_artists", []))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _apply_filters(df: pd.DataFrame, filters: Optional[FeatureFilters]) -> pd.DataFrame:
        if not filters:
            return df
        mask = pd.Series(True, index=df.index)
        for key, col in (("energy_range", "energy"), ("valence_range", "valence"), ("bpm_range", "tempo")):
            if key in filters:
                lo, hi = filters[key]
                mask &= df[col].between(lo, hi)
        if filters.get("danceability_min") is not None:
            mask &= df["danceability"] >= filters["danceability_min"]
        if "year_range" in filters and "year" in df:
            lo, hi = filters["year_range"]
            mask &= df["year"].between(lo, hi)
        return df[mask]


def load_profile(name: str) -> Dict[str, Any]:
    """Load an offline listener profile from profiles/<name>.json."""
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        logger.warning(f"Profile '{name}' not found at {path}; using an empty taste profile")
        return {"name": name, "seed_artists": [], "seed_track_ids": []}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("name", name)
    data.setdefault("seed_artists", [])
    data.setdefault("seed_track_ids", [])
    return data
