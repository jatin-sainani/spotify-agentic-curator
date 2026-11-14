"""Deterministic playlist metrics against gold prompt expectations.

Every metric is in [0, 1] (higher is better) unless noted, and is only computed
when the gold prompt defines the expectation it needs; otherwise it is None and
excluded from averages.
"""

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.agent.ranker import FEATURE_TOLERANCE, _range_fit, song_key
from app.core.draft_models import TrackRef

TOLERANCE = {**FEATURE_TOLERANCE, "acousticness": 0.25, "instrumentalness": 0.25}

# Metrics that make up the gold score: did the playlist deliver what the prompt asked?
GOLD_COMPONENTS = ("genre_hit_rate", "era_hit_rate", "target_fit", "avoid_clean_rate", "arc_score_01")


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (None if undefined, e.g. constant input)."""
    return pearson(_ranks(x), _ranks(y))


def normalized_entropy(labels: Iterable[str]) -> Optional[float]:
    counts = Counter(labels)
    n = sum(counts.values())
    if n == 0:
        return None
    if len(counts) == 1:
        return 0.0
    h = -sum(c / n * math.log(c / n) for c in counts.values())
    return h / math.log(min(n, len(counts)))


# ---------------------------------------------------------------------------
# Arc shapes
# ---------------------------------------------------------------------------

def arc_curve(shape: str, n: int) -> List[float]:
    """Expected relative energy at each position for an arc shape."""
    if n <= 1:
        return [0.0] * n
    xs = [i / (n - 1) for i in range(n)]
    if shape == "build":
        return xs
    if shape == "cooldown":
        return [1 - x for x in xs]
    if shape == "peak":
        return [1 - abs(2 * x - 1) for x in xs]
    if shape == "wave":
        return [math.sin(2 * math.pi * x) for x in xs]
    raise ValueError(f"Unknown arc shape '{shape}'")


# ---------------------------------------------------------------------------
# Per-playlist metrics
# ---------------------------------------------------------------------------

def _text(track: TrackRef) -> str:
    return " ".join([*track.genres, track.name]).lower()


def _genre_text(track: TrackRef) -> str:
    return " ".join(track.genres).lower()


def playlist_metrics(
    tracks: List[TrackRef],
    gold: Dict[str, Any],
    length: int,
    seed_artists: Iterable[str] = (),
) -> Dict[str, Optional[float]]:
    """Compute all applicable metrics for one playlist."""
    n = len(tracks)
    m: Dict[str, Optional[float]] = {"n_tracks": float(n), "fill_rate": min(1.0, n / length) if length else None}
    if n == 0:
        m["gold_score"] = 0.0
        return m

    # -- gold expectations -------------------------------------------------
    if gold.get("genres_any"):
        keys = [k.lower() for k in gold["genres_any"]]
        m["genre_hit_rate"] = sum(any(k in _genre_text(t) for k in keys) for t in tracks) / n
    if gold.get("avoid"):
        keys = [k.lower() for k in gold["avoid"]]
        m["avoid_clean_rate"] = sum(not any(k in _text(t) for k in keys) for t in tracks) / n
    if gold.get("era"):
        lo, hi = gold["era"]
        dated = [t for t in tracks if t.year]
        m["era_hit_rate"] = (sum(lo <= t.year <= hi for t in dated) / len(dated)) if dated else 0.0
    if gold.get("targets"):
        fits, hits = [], []
        for t in tracks:
            f = t.audio_features or {}
            per = [_range_fit(float(f[k]), lo, hi, TOLERANCE.get(k, 0.25))
                   for k, (lo, hi) in gold["targets"].items() if k in f]
            if per:
                fits.append(sum(per) / len(per))
                hits.append(all(p == 1.0 for p in per))
        m["target_fit"] = sum(fits) / len(fits) if fits else 0.0
        m["target_hit_rate"] = sum(hits) / len(hits) if hits else 0.0
    if gold.get("arc"):
        energies = [t.audio_features["energy"] for t in tracks if t.audio_features and "energy" in t.audio_features]
        rho = spearman(energies, arc_curve(gold["arc"], len(energies)))
        m["arc_score"] = rho if rho is not None else 0.0  # in [-1, 1]
        m["arc_score_01"] = (m["arc_score"] + 1) / 2

    # -- list quality ----------------------------------------------------------
    primaries = [t.artists[0] if t.artists else "" for t in tracks]
    counts = Counter(primaries)
    m["artist_diversity"] = len(counts) / n
    m["max_artist_share"] = max(counts.values()) / n  # lower is better
    m["genre_entropy"] = normalized_entropy(
        (t.genres[1] if len(t.genres) > 1 else (t.genres[0] if t.genres else "?")) for t in tracks)
    m["duplicate_songs"] = float(n - len({song_key(t) for t in tracks}))  # lower is better
    seeds = {a.lower() for a in seed_artists}
    m["familiarity"] = sum(p.lower() in seeds for p in primaries) / n if seeds else 0.0  # descriptive

    parts = [m[k] for k in GOLD_COMPONENTS if m.get(k) is not None]
    m["gold_score"] = sum(parts) / len(parts) if parts else None
    return m
