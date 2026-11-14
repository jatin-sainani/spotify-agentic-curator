"""Judge-vs-human agreement: is the LLM judge trustworthy?

Workflow:
  1. python -m app eval label <results.json>      -> evals/human_labels/<run>.csv
  2. Fill the empty `label` column with 0, 0.5 or 1 (same rubric as the judge).
  3. python -m app eval agreement <results.json> <labels.csv>

Reports exact agreement, linearly weighted Cohen's kappa on the 3 grades, and
Spearman correlation between judge and human labels.
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evals.judge import GRADES, judge_track_line, snap
from evals.metrics import spearman

LABEL_FIELDS = ["run", "prompt_id", "config", "position", "track_id", "request", "track", "label"]


def weighted_kappa(a: Sequence[float], b: Sequence[float], categories: Sequence[float] = GRADES) -> Optional[float]:
    """Linearly weighted Cohen's kappa for ordinal labels."""
    if len(a) != len(b) or not a:
        return None
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1
    n = len(a)
    row = [sum(observed[i]) / n for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) / n for j in range(k)]
    weight = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs = sum(weight[i][j] * observed[i][j] / n for i in range(k) for j in range(k))
    exp = sum(weight[i][j] * row[i] * col[j] for i in range(k) for j in range(k))
    if exp == 0:
        return 1.0 if obs == 0 else None
    return 1 - obs / exp


def write_label_sheet(results: Dict, out_path: Path, n: int = 40, seed: int = 7) -> int:
    """Sample judged tracks (stratified across configs) into a CSV to hand-label.

    Judge labels are deliberately not included, so labeling is blind.
    """
    from app.core.draft_models import TrackRef

    rows = []
    for rec in results["records"]:
        for pos, t in enumerate(rec["tracks"]):
            ref = TrackRef(**t)
            rows.append({
                "run": results["run"], "prompt_id": rec["prompt_id"], "config": rec["config"],
                "position": pos, "track_id": ref.id, "request": rec["prompt"],
                "track": judge_track_line(pos + 1, ref).split(". ", 1)[1], "label": "",
            })
    rng = random.Random(seed)
    by_config: Dict[str, List[Dict]] = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r)
    for group in by_config.values():
        rng.shuffle(group)
    sample: List[Dict] = []
    while len(sample) < n and any(by_config.values()):
        for group in by_config.values():
            if group and len(sample) < n:
                sample.append(group.pop())
    rng.shuffle(sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    return len(sample)


def load_labels(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if str(r.get("label", "")).strip() != ""]


def pair_labels(results: Dict, labels: List[Dict]) -> List[Tuple[float, float]]:
    """(judge, human) label pairs; judge averages across passes are snapped to a grade."""
    judged = {}
    for rec in results["records"]:
        fits = (rec.get("judge") or {}).get("track_fits") or []
        for pos, fit in enumerate(fits):
            judged[(rec["prompt_id"], rec["config"], pos)] = fit
    pairs = []
    for row in labels:
        human = snap(row["label"])
        judge = judged.get((row["prompt_id"], row["config"], int(row["position"])))
        if human is not None and judge is not None:
            pairs.append((snap(judge), human))
    return pairs


def agreement_report(pairs: List[Tuple[float, float]]) -> Dict[str, Optional[float]]:
    if not pairs:
        return {"n": 0, "exact_agreement": None, "weighted_kappa": None, "spearman": None}
    judge, human = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "exact_agreement": sum(a == b for a, b in pairs) / len(pairs),
        "weighted_kappa": weighted_kappa(judge, human),
        "spearman": spearman(judge, human),
        "judge_mean": sum(judge) / len(judge),
        "human_mean": sum(human) / len(human),
    }
