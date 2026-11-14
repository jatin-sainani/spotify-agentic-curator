"""Run the injection suite and report attack success rate (ASR) per defense mode.

    python -m app eval injection --modes off,prompt_only,all_on
    python -m app eval injection --attacks score_inflation --name inj   # resumable chunks
    python -m app eval injection --fpr-csv songs.csv                      # false positives on real data
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging import logger
from evals.injection.scenarios import CASES, DEFENSE_MODES, run_case

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def false_positive_rate(csv_path: str) -> Dict[str, Any]:
    """How often the detector flags real catalog metadata (would wrongly drop tracks)."""
    import pandas as pd

    from app.agent.safety import _INJECTION_RE

    path = Path(csv_path)
    cols = ["name", "artists", "niche_genres"]
    df = pd.read_parquet(path, columns=cols) if path.suffix == ".parquet" else pd.read_csv(path, usecols=cols)
    out: Dict[str, Any] = {"source": path.name, "tracks": len(df), "fields": {}}
    flagged_rows = pd.Series(False, index=df.index)
    for col in cols:
        hits = df[col].fillna("").astype(str).str.contains(_INJECTION_RE)
        flagged_rows |= hits
        out["fields"][col] = int(hits.sum())
    out["flagged_tracks"] = int(flagged_rows.sum())
    out["rate"] = out["flagged_tracks"] / max(1, len(df))
    out["examples"] = df.loc[flagged_rows, "name"].astype(str).head(10).tolist()
    return out


def run_suite(
    name: str,
    attacks: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
    out_dir: Path = RESULTS_DIR,
    meta: Optional[Dict[str, Any]] = None,
    **llms,
) -> Dict[str, Any]:
    modes = modes or list(DEFENSE_MODES)
    cases = [c for c in CASES if not attacks or c.attack in attacks]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "run": name, "started": datetime.now().isoformat(timespec="seconds"), "meta": meta or {}, "records": []}
    done = {(r["case"], r["mode"]) for r in results["records"]}

    for case in cases:
        for mode in modes:
            if (case.id, mode) in done:
                continue
            logger.info(f"[injection] {case.id} x {mode}")
            try:
                outcome = run_case(case, mode, **llms)
                error = None
            except Exception as e:  # a crash is recorded, never counted as a defense success
                logger.error(f"[injection] {case.id} x {mode} failed: {e}")
                outcome, error = {"success": None}, str(e)
            results["records"].append({"case": case.id, "attack": case.attack, "variant": case.variant,
                                       "mode": mode, "error": error, **outcome})
            path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return results


def build_report(results: Dict[str, Any]) -> str:
    recs = results["records"]
    modes = [m for m in DEFENSE_MODES if any(r["mode"] == m for r in recs)]
    attacks = [a for a in dict.fromkeys(c.attack for c in CASES) if any(r["attack"] == a for r in recs)]
    meta = results.get("meta", {})

    lines = [f"# Prompt-injection red team: `{results['run']}`", "",
             f"Model: `{meta.get('model', '?')}`. Each cell: successful attacks / attempts "
             "(lower is better). Payloads live in catalog metadata (track titles).", "",
             "| Attack | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    totals = defaultdict(lambda: [0, 0])
    for attack in attacks:
        cells = []
        for mode in modes:
            rs = [r for r in recs if r["attack"] == attack and r["mode"] == mode and r.get("success") is not None]
            wins = sum(bool(r["success"]) for r in rs)
            totals[mode][0] += wins
            totals[mode][1] += len(rs)
            cells.append(f"{wins}/{len(rs)}" if rs else "–")
        lines.append(f"| {attack} | " + " | ".join(cells) + " |")
    # Agent attacks only test something if the agent's searches returned the carrier.
    for attack in [a for a in attacks if a in ("agent_hijack", "id_injection")]:
        cells = []
        for mode in modes:
            rs = [r for r in recs if r["attack"] == attack and r["mode"] == mode and r.get("success") is not None]
            exposed = [r for r in rs if r.get("exposed")]
            cells.append(f"{sum(bool(r['success']) for r in exposed)}/{len(exposed)} exposed" if rs else "–")
        lines.append(f"| ↳ {attack}, payload seen by agent | " + " | ".join(cells) + " |")
    lines.append("| **all attacks (ASR)** | " + " | ".join(
        f"**{totals[m][0]}/{totals[m][1]} = {totals[m][0] / totals[m][1]:.0%}**" if totals[m][1] else "–"
        for m in modes) + " |")

    lines += ["", "Defense modes: `off` = no defenses; `prompt_only` = metadata sanitization + "
              "`<catalog_data>` spotlighting, flagged tracks still eligible; `all_on` = also drop flagged "
              "tracks before ranking. Structural guarantees (read-only tools, selection only from "
              "tool-returned tracks, per-artist cap) are always on.", ""]

    errors = [r for r in recs if r.get("error")]
    if errors:
        lines.append(f"**{len(errors)} runs errored** and are excluded: "
                     + ", ".join(f"{r['case']}×{r['mode']}" for r in errors))
        lines.append("")

    lines += ["## Per case", "", "| Case | Mode | Success | Detail |", "|---|---|---|---|"]
    for r in recs:
        if r["attack"] == "score_inflation":
            rel = r.get("attack_relevance")
            detail = (f"attack relevance {rel if rel is not None else 'n/a (dropped before scoring)'} "
                      f"vs control {r.get('control_relevance')}; in top-3: {r.get('success')}")
        elif r["attack"] in ("agent_hijack", "id_injection"):
            detail = (f"payload seen: {r.get('exposed')}; searches: "
                      + "; ".join(q for q in (r.get("queries") or []) if q)[:120])
            if r["attack"] == "agent_hijack":
                detail += f" (promo search: {r.get('searched_promo')}, promo in playlist: {r.get('promo_in_playlist')})"
        else:
            detail = (f"precision {r.get('precision', 0):.2f} vs control {r.get('control_precision', 0):.2f} "
                      f"(lift {r.get('lift', 0):+.2f})") if r.get("precision") is not None else ""
        status = "error" if r.get("error") else ("**yes**" if r.get("success") else "no")
        lines.append(f"| {r['case']} | {r['mode']} | {status} | {detail} |")

    fpr = results.get("false_positives")
    if fpr:
        lines += ["", "## False positives on real catalog metadata", "",
                  f"The detector flagged **{fpr['flagged_tracks']} of {fpr['tracks']:,} tracks "
                  f"({fpr['rate']:.4%})** in `{fpr['source']}` (titles: {fpr['fields']['name']}, "
                  f"artists: {fpr['fields']['artists']}, genres: {fpr['fields']['niche_genres']}). "
                  f"With `drop_flagged`, these tracks are never recommended. Examples: "
                  + ", ".join(f"\"{e}\"" for e in fpr["examples"])]
    lines.append("")
    return "\n".join(lines)
