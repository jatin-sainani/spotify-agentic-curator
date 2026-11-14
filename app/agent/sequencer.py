"""Order a playlist along an energy arc (e.g. warmup -> peak -> cooldown).

Selection decides *which* tracks are in the playlist; sequencing decides their
order. Given phases like [{"name": "warmup", "pct": 0.2, "energy_target": 0.4}, ...],
each slot gets a target energy, then tracks are matched to slots by rank (lowest
energy to lowest target, and so on). In one dimension, rank matching minimizes the
total |energy - target| distance, so this is optimal and needs no solver.
"""

import re
from typing import Any, Dict, List, Optional

from app.core.draft_models import TrackRef

ARC_PHASES = {
    "build": [{"name": "low", "pct": 0.3, "energy_target": 0.35},
              {"name": "rise", "pct": 0.4, "energy_target": 0.6},
              {"name": "high", "pct": 0.3, "energy_target": 0.85}],
    "cooldown": [{"name": "high", "pct": 0.3, "energy_target": 0.8},
                 {"name": "ease", "pct": 0.4, "energy_target": 0.55},
                 {"name": "low", "pct": 0.3, "energy_target": 0.25}],
    "peak": [{"name": "warmup", "pct": 0.25, "energy_target": 0.45},
             {"name": "peak", "pct": 0.5, "energy_target": 0.85},
             {"name": "landing", "pct": 0.25, "energy_target": 0.3}],
}

# Checked in order: "peaks in the middle, then cools down" is a peak, not a cooldown.
_ARC_PATTERNS = [
    ("peak", r"\bpeaks?\b|\bin the middle\b|rises?,? (?:and )?peaks?|warm ?up.*cool ?down|lands? softly"),
    ("cooldown", r"wind[- ]?down|comedown|cool(?:s|ing)? down|loud to quiet|upbeat to (?:sleepy|calm|chill)"
                 r"|winds? down|to sleepy|getting (?:calmer|quieter)"),
    ("build", r"\bbuilds?\b|building|ramp(?:s|ing)? up|keeps? ramping|starts? (?:mellow|slow|calm|quiet|soft)"
              r"|slowly wake|toward(?:s)? (?:a |the )?(?:sprint|finish|end|peak)|gets? (?:louder|faster)"),
]


def parse_arc(text: str) -> Optional[str]:
    """Energy-arc shape the request describes ("build", "cooldown", "peak"), or None."""
    text = (text or "").lower()
    for shape, pattern in _ARC_PATTERNS:
        if re.search(pattern, text):
            return shape
    return None


def slot_targets(phases: List[Dict[str, Any]], n: int) -> List[float]:
    """Target energy for each of n positions, following the phases' shares."""
    if not phases or n <= 0:
        return []
    counts = [max(0, round(float(p.get("pct", 1 / len(phases))) * n)) for p in phases]
    # Fix rounding so counts sum to n (adjust the largest phase).
    diff = n - sum(counts)
    counts[counts.index(max(counts))] += diff
    targets: List[float] = []
    for phase, count in zip(phases, counts):
        targets.extend([float(phase.get("energy_target", 0.5))] * max(0, count))
    return targets[:n]


def sequence_by_energy(tracks: List[TrackRef], phases: List[Dict[str, Any]]) -> List[TrackRef]:
    """Reorder tracks to follow the phases' energy targets.

    Tracks without an energy feature keep their relative order and fill the
    remaining positions at the end.
    """
    with_energy = [t for t in tracks if t.audio_features and "energy" in t.audio_features]
    without = [t for t in tracks if t not in with_energy]
    targets = slot_targets(phases, len(with_energy))
    if len(targets) != len(with_energy) or len(set(targets)) <= 1:
        return list(tracks)  # flat or missing arc: nothing to sequence

    slots_by_target = sorted(range(len(targets)), key=lambda i: (targets[i], i))
    tracks_by_energy = sorted(with_energy, key=lambda t: t.audio_features["energy"])
    ordered: List[TrackRef] = [None] * len(targets)  # type: ignore[list-item]
    for slot, track in zip(slots_by_target, tracks_by_energy):
        ordered[slot] = track
    return ordered + without
