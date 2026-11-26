"""Summarize completed behavior runs using matched user/topic/seed probes."""

import argparse
import json
from pathlib import Path


def summarize(path):
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    rows = json.loads((path / "rows.json").read_text(encoding="utf-8"))
    responses = json.loads((path / "responses.json").read_text(encoding="utf-8"))

    def key(row):
        return tuple(row[k] for k in ("mode", "user", "phase", "kind", "seed"))

    metrics = {key(row): row for row in rows}
    cards = {key(row): [x["client_id"] for x in row["response"]["items"]] for row in responses}
    output = []
    for stage in ("after", "final"):
        for user in sorted({r["user"] for r in rows}):
            for kind in ("feed", "heldout", "explicit_search"):
                group = [
                    r
                    for r in rows
                    if r["mode"] == "on"
                    and r["user"] == user
                    and r["kind"] == kind
                    and r["phase"].startswith(stage + "_")
                ]
                if not group:
                    continue
                paired = []
                order_changes = membership_changes = 0
                for row in group:
                    cold = ("on", user, "cold_" + row["focus"], kind, row["seed"])
                    baseline = ("on_semantic", user, row["phase"], kind, row["seed"])
                    paired.append((metrics[cold], row, metrics[baseline]))
                    order_changes += cards[cold] != cards[key(row)]
                    membership_changes += set(cards[cold]) != set(cards[key(row)])
                result = {
                    "stage": stage,
                    "user": user,
                    "kind": kind,
                    "pairs": len(paired),
                    "order_changes": order_changes,
                    "membership_changes": membership_changes,
                }
                for i, name in enumerate(("cold", "trained", "baseline")):
                    selected = [p[i] for p in paired]
                    returned = sum(r["returned"] for r in selected)
                    result[name] = {
                        "precision": sum(r["relevant"] for r in selected) / returned
                        if returned
                        else None,
                        "focus_ndcg10": sum(r["focus_ndcg10"] for r in selected) / len(selected),
                        "focus_count": sum(r["focus_count"] for r in selected) / len(selected),
                        "needs_covered": sum(r["needs_covered"] for r in selected) / len(selected),
                    }
                result["ndcg_delta_vs_cold"] = (
                    result["trained"]["focus_ndcg10"] - result["cold"]["focus_ndcg10"]
                )
                result["ndcg_delta_vs_baseline"] = (
                    result["trained"]["focus_ndcg10"] - result["baseline"]["focus_ndcg10"]
                )
                output.append(result)
    result = {
        "requests": report["requests"],
        "checks_pass": report["checks_pass"],
        "event_counts": report["event_counts"],
        "paired_metrics": output,
        "limitations": report["limitations"],
    }
    (path / "paired-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    summarize(parser.parse_args().path)
