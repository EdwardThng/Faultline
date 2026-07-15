"""Full-sweep execution from a sweep config.

A sweep runs, per model x task x seed, one fault-free baseline plus every
applicable fault spec, and writes all traces to the runs directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from faultline.harness.faults import FaultSpec
from faultline.harness.runner import fault_grid, run_one
from faultline.tasks.loader import load_task

DEFAULT_MODES = ["hard_error", "transient", "malformed", "empty", "timeout"]
DEFAULT_VARIANTS = ["caught", "propagated"]


def run_sweep(config_path: str | Path, verbose: bool = True) -> list:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    models = cfg["models"]
    task_paths = sorted(Path().glob(cfg.get("tasks", "tasks/*.yaml")))
    seeds = cfg.get("seeds", [0])
    runs_dir = cfg.get("runs_dir", "runs")
    specs = fault_grid(
        cfg.get("fault_modes", DEFAULT_MODES),
        cfg.get("variants", DEFAULT_VARIANTS),
    )

    traces = []
    for model in models:
        for task_path in task_paths:
            task = load_task(task_path)
            for seed in seeds:
                for spec in [FaultSpec.none(), *specs]:
                    trace = run_one(
                        task, model=model, fault=spec, seed=seed,
                        runs_dir=runs_dir,
                    )
                    traces.append(trace)
                    if verbose:
                        label = spec.mode.value
                        if spec.mode.value != "none":
                            label += f"/{spec.variant.value}"
                        print(
                            f"[{model}] {task.name} seed={seed} "
                            f"fault={label} -> {trace.termination}"
                            f"{' OK' if trace.success else ' FAIL'}"
                        )
    return traces
