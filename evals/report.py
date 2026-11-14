"""Summarize an eval results file into a Markdown report."""

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evals.judge import DIMENSIONS

HEADLINE = [
    ("gold_score", "Gold score", True),
    ("judge_precision", "Judge precision", True),
    ("judge_intent_alignment", "Intent alignment", True),
    ("genre_hit_rate", "Genre hit", True),
    ("era_hit_rate", "Era hit", True),
    ("target_fit", "Feature fit", True),
    ("avoid_clean_rate", "Exclusions honored", True),
    ("arc_score", "Arc (Spearman)", True),
    ("artist_diversity", "Artist diversity", True),
    ("duplicate_songs", "Duplicate songs", False),
    ("familiarity", "Familiar artists", None),
    ("curate_s", "Latency (s)", False),
]


def flatten(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """One flat dict of every numeric measure for a record."""
    flat = dict(record.get("metrics") or {})
    judge = record.get("judge") or {}
    flat["judge_precision"] = judge.get("precision")
    for d in DIMENSIONS:
        flat[f"judge_{d}"] = (judge.get("dims") or {}).get(d)
    flat["judge_self_agreement"] = judge.get("self_agreement")
    flat["curate_s"] = (record.get("timing") or {}).get("curate_s")
    flat["tool_calls"] = float(len(record.get("tool_calls") or []))
    return flat


def mean_se(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None, len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var / len(vals)), len(vals)


def paired_diff(a: Dict[str, float], b: Dict[str, float], iters: int = 2000, seed: int = 0):
    """Mean of (b - a) over shared prompts, a 95% bootstrap CI, and b's win rate."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if not keys:
        return None
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    wins = sum(d > 1e-9 for d in diffs) / len(diffs)
    return {"mean": sum(diffs) / len(diffs), "lo": boots[int(0.025 * iters)], "hi": boots[int(0.975 * iters)],
            "win_rate": wins, "n": len(diffs)}


def fmt(v: Optional[float], digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def build_report(results: Dict[str, Any], baseline: str = "baseline") -> str:
    records = results["records"]
    configs = [c for c in results["configs"] if any(r["config"] == c for r in records)]
    by_config: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)
    prompts = sorted({r["prompt_id"] for r in records})
    meta = results.get("meta", {})

    lines = [f"# Eval report: `{results['run']}`", ""]
    lines.append(f"{len(prompts)} prompts × {len(configs)} configs = {len(records)} playlists. "
                 f"Agent/scorer: `{meta.get('agent_model', '?')}`, judge: `{meta.get('judge_model', 'none')}`, "
                 f"catalog: `{meta.get('catalog', '?')}`.")
    errors = [r for r in records if r.get("error")]
    if errors:
        lines.append(f"\n**{len(errors)} runs errored** (scored as empty playlists): "
                     + ", ".join(f"{r['prompt_id']}×{r['config']}" for r in errors[:10]))
    lines += ["", "## Configs", ""]
    for c in configs:
        lines.append(f"- **{c}**: {results['configs'][c].get('description', '')}")

    # -- headline table ------------------------------------------------------
    lines += ["", "## Results (mean ± standard error)", ""]
    lines.append("| Metric | " + " | ".join(configs) + " |")
    lines.append("|---|" + "---|" * len(configs))
    for key, label, higher in HEADLINE:
        cells, any_value = [], False
        for c in configs:
            m, se, n = mean_se([flatten(r).get(key) for r in by_config[c]])
            any_value |= m is not None
            cells.append("–" if m is None else (f"{m:.2f} ± {se:.2f}" if se is not None else f"{m:.2f}"))
        if any_value:
            arrow = "" if higher is None else (" ↑" if higher else " ↓")
            lines.append(f"| {label}{arrow} | " + " | ".join(cells) + " |")

    # -- paired comparison vs baseline ------------------------------------------
    if baseline in configs and len(configs) > 1:
        lines += ["", f"## Paired difference vs `{baseline}` (95% bootstrap CI, per-prompt win rate)", ""]
        lines.append("| Config | Gold score Δ | Judge precision Δ | Intent alignment Δ |")
        lines.append("|---|---|---|---|")
        base = {r["prompt_id"]: flatten(r) for r in by_config[baseline]}
        for c in configs:
            if c == baseline:
                continue
            other = {r["prompt_id"]: flatten(r) for r in by_config[c]}
            cells = []
            for key in ("gold_score", "judge_precision", "judge_intent_alignment"):
                d = paired_diff({k: v.get(key) for k, v in base.items()}, {k: v.get(key) for k, v in other.items()})
                cells.append("–" if d is None else
                             f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}], wins {d['win_rate']:.0%}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")

    # -- per category -------------------------------------------------------------
    cats = sorted({r["category"] for r in records})
    lines += ["", "## Gold score by category", ""]
    lines.append("| Category | n | " + " | ".join(configs) + " |")
    lines.append("|---|---|" + "---|" * len(configs))
    for cat in cats:
        n = len({r["prompt_id"] for r in records if r["category"] == cat})
        cells = [fmt(mean_se([flatten(r).get("gold_score") for r in by_config[c] if r["category"] == cat])[0])
                 for c in configs]
        lines.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")

    # -- judge diagnostics ------------------------------------------------------------
    judged = [r for r in records if r.get("judge")]
    if judged:
        lines += ["", "## Judge diagnostics", ""]
        agree = mean_se([flatten(r).get("judge_self_agreement") for r in judged])[0]
        failed = sum((r["judge"] or {}).get("failed_passes", 0) for r in judged)
        lines.append(f"- Self-consistency (same label in original vs shuffled order): **{fmt(agree)}** "
                     f"over {len(judged)} playlists; {failed} failed passes.")
        gold = [flatten(r).get("gold_score") for r in judged]
        prec = [flatten(r).get("judge_precision") for r in judged]
        from evals.metrics import spearman
        pairs = [(g, p) for g, p in zip(gold, prec) if g is not None and p is not None]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else None
        lines.append(f"- Spearman(judge precision, deterministic gold score) across playlists: **{fmt(rho)}** "
                     f"(n={len(pairs)}): do the judge and the hand-written expectations agree?")
        lines.append("- Judge-vs-human agreement: see `python -m app eval agreement` once labels exist.")

    # -- notable failures -----------------------------------------------------------------
    worst = sorted((r for r in records if flatten(r).get("gold_score") is not None),
                   key=lambda r: flatten(r)["gold_score"])[:5]
    if worst:
        lines += ["", "## Lowest gold scores", ""]
        for r in worst:
            names = "; ".join(f"{t['name']} ({', '.join(t['artists'][:1])})" for t in r["tracks"][:4])
            summary = next((s for s in ((r.get("judge") or {}).get("summaries") or []) if s), "")
            lines.append(f"- `{r['prompt_id']}` × `{r['config']}` ({fmt(flatten(r)['gold_score'])}): "
                         f"*{r['prompt']}* → {names}…" + (f" Judge: {summary}" if summary else ""))
    lines.append("")
    return "\n".join(lines)
