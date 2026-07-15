"""Leaderboard data generation.

Writes ``leaderboard.json`` plus per-run trace files into the site data
directory, so the static leaderboard is self-contained and every cell can
link down to the individual tool calls that produced it.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from faultline.analysis.aggregate import aggregate_runs, load_runs
from faultline.telemetry.verdicts import VerdictThresholds

DEFAULT_CONFIG = "analysis/config.yaml"


def generate_report(
    runs_dir: str | Path = "runs",
    out_dir: str | Path = "site/data",
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    thresholds = (
        VerdictThresholds.from_config(config_path)
        if Path(config_path).exists()
        else VerdictThresholds()
    )
    traces = load_runs(runs_dir)
    if not traces:
        raise SystemExit(f"no run traces found in {runs_dir}")

    board = aggregate_runs(traces, thresholds)
    board["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board["thresholds"] = thresholds.__dict__
    board["total_runs"] = len(traces)

    out = Path(out_dir)
    trace_out = out / "traces"
    trace_out.mkdir(parents=True, exist_ok=True)
    for p in Path(runs_dir).glob("*.json"):
        shutil.copy2(p, trace_out / p.name)

    target = out / "leaderboard.json"
    target.write_text(json.dumps(board, indent=2), encoding="utf-8")
    return target
