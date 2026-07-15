"""Aggregate run traces into leaderboard data.

Every faulted run is paired with the fault-free baseline of the same
(model, task, seed) — like against like — classified by the verdict
pipeline, and rolled up into a model x failure-mode matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faultline.analysis.stats import median, wilson_interval
from faultline.telemetry.trace import RunTrace
from faultline.telemetry.verdicts import VerdictThresholds, classify

VERDICTS = ["recovered", "degraded", "spiraled", "doom_looped", "crashed"]


def load_runs(runs_dir: str | Path) -> list[RunTrace]:
    return [RunTrace.load(p) for p in sorted(Path(runs_dir).glob("*.json"))]


def _fault_key(trace: RunTrace) -> str:
    mode = trace.fault.get("mode", "none")
    if mode == "none":
        return "none"
    from faultline.harness.faults import FaultMode, variant_applies

    if variant_applies(FaultMode(mode)):
        return f"{mode}/{trace.fault.get('variant', 'caught')}"
    return mode


def classify_all(
    traces: list[RunTrace], thresholds: VerdictThresholds | None = None
) -> list[dict[str, Any]]:
    """Pair faulted runs with their baselines and classify each one."""
    baselines: dict[tuple, RunTrace] = {}
    for t in traces:
        if t.is_baseline:
            baselines[(t.model, t.task, t.seed)] = t

    records = []
    for t in traces:
        if t.is_baseline:
            continue
        baseline = baselines.get((t.model, t.task, t.seed))
        if baseline is None:
            raise ValueError(
                f"no baseline for run {t.run_id} "
                f"({t.model}/{t.task}/seed {t.seed}); re-run the sweep"
            )
        record = classify(t, baseline, thresholds)
        record["fault_key"] = _fault_key(t)
        records.append(record)
    return records


def aggregate_runs(
    traces: list[RunTrace], thresholds: VerdictThresholds | None = None
) -> dict[str, Any]:
    """Roll classified runs up into the model x failure-mode matrix."""
    records = classify_all(traces, thresholds)

    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        cells.setdefault((r["model"], r["fault_key"]), []).append(r)

    cell_rows = []
    for (model, fault_key), rs in sorted(cells.items()):
        n = len(rs)
        recovered = sum(r["verdict"] == "recovered" for r in rs)
        doomed = [r for r in rs if r["verdict"] == "doom_looped"]
        recovered_rs = [r for r in rs if r["verdict"] == "recovered"]
        low, high = wilson_interval(recovered, n)
        cell_rows.append(
            {
                "model": model,
                "fault_key": fault_key,
                "mode": fault_key.split("/")[0],
                "variant": (
                    fault_key.split("/")[1] if "/" in fault_key else "n/a"
                ),
                "n": n,
                "recovery_rate": recovered / n,
                "ci_low": low,
                "ci_high": high,
                "verdicts": {
                    v: sum(r["verdict"] == v for r in rs) for v in VERDICTS
                },
                "doom_loop_rate": len(doomed) / n,
                "median_loop_len": median(
                    [r["max_signature_repeats"] for r in doomed]
                ),
                "recovery_overhead_steps": median(
                    [r["extra_steps"] for r in recovered_rs]
                ),
                "recovery_overhead_tokens": median(
                    [r["extra_tokens"] for r in recovered_rs]
                ),
                "run_ids": [r["run_id"] for r in rs],
            }
        )

    # caught vs propagated delta: how much of the failure rate is the
    # model vs the loop, per exception-shaped mode
    deltas = []
    by_mv: dict[tuple[str, str, str], dict] = {
        (c["model"], c["mode"], c["variant"]): c for c in cell_rows
    }
    for (model, mode, variant), cell in sorted(by_mv.items()):
        if variant != "caught":
            continue
        prop = by_mv.get((model, mode, "propagated"))
        if prop:
            deltas.append(
                {
                    "model": model,
                    "mode": mode,
                    "caught_recovery": cell["recovery_rate"],
                    "propagated_recovery": prop["recovery_rate"],
                    "delta": cell["recovery_rate"] - prop["recovery_rate"],
                }
            )

    return {
        "models": sorted({c["model"] for c in cell_rows}),
        "fault_keys": sorted({c["fault_key"] for c in cell_rows}),
        "tasks": sorted({r["task"] for r in records}),
        "cells": cell_rows,
        "caught_vs_propagated": deltas,
        "runs": {r["run_id"]: r for r in records},
    }
