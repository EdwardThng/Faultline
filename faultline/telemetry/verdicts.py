"""The verdict pipeline.

Every faulted run is scored against a fault-free baseline of the same task
and model, then classified purely from trace telemetry — step counts,
repeated tool-call signatures, token cost, termination state. No LLM judge:
every classification is reproducible from the trace.

Classification order (first match wins):

1. Crashed      — the loop terminated abnormally.
2. Recovered    — task succeeded within bounded overhead of baseline.
3. Degraded     — task succeeded, but at excessive cost.
4. Doom-looped  — task failed after repeating one tool-call signature past
                  the threshold.
5. Spiraled     — task failed any other way: the agent abandoned the task
                  strategy (wandered into novel-but-unproductive calls, gave
                  up, or hit the step ceiling without looping).

All thresholds live in ``analysis/config.yaml`` and are versioned with the
results they produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from faultline.telemetry.trace import RunTrace


class Verdict(str, Enum):
    RECOVERED = "recovered"
    DEGRADED = "degraded"
    SPIRALED = "spiraled"
    DOOM_LOOPED = "doom_looped"
    CRASHED = "crashed"


@dataclass(frozen=True)
class VerdictThresholds:
    #: A failed run with any single signature repeated at least this many
    #: times is a doom loop.
    doom_loop_repeats: int = 5
    #: Recovered allows steps up to baseline * ratio + abs slack.
    recovered_max_step_overhead_ratio: float = 1.5
    recovered_max_step_overhead_abs: int = 6
    #: Recovered allows total tokens up to baseline * ratio (skipped when
    #: the baseline recorded no token counts).
    recovered_max_token_overhead_ratio: float = 2.5

    @classmethod
    def from_config(cls, path: str | Path) -> "VerdictThresholds":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data.get("verdict_thresholds", {}))


def step_budget(baseline_steps: int, t: VerdictThresholds) -> float:
    return (
        baseline_steps * t.recovered_max_step_overhead_ratio
        + t.recovered_max_step_overhead_abs
    )


def classify(
    faulted: RunTrace,
    baseline: RunTrace,
    thresholds: VerdictThresholds | None = None,
) -> dict[str, Any]:
    """Classify one faulted run against its fault-free baseline.

    Returns the verdict plus the derived metrics it was based on, so the
    decision is inspectable next to the number it produced.
    """
    t = thresholds or VerdictThresholds()
    repeats = faulted.max_signature_repeats()
    extra_steps = faulted.steps - baseline.steps
    faulted_tokens = faulted.tokens_in + faulted.tokens_out
    baseline_tokens = baseline.tokens_in + baseline.tokens_out

    if faulted.termination == "crashed":
        verdict = Verdict.CRASHED
    elif faulted.success:
        within_steps = faulted.steps <= step_budget(baseline.steps, t)
        within_tokens = (
            baseline_tokens == 0
            or faulted_tokens
            <= baseline_tokens * t.recovered_max_token_overhead_ratio
        )
        verdict = (
            Verdict.RECOVERED if within_steps and within_tokens
            else Verdict.DEGRADED
        )
    elif repeats >= t.doom_loop_repeats:
        verdict = Verdict.DOOM_LOOPED
    else:
        verdict = Verdict.SPIRALED

    return {
        "run_id": faulted.run_id,
        "baseline_run_id": baseline.run_id,
        "model": faulted.model,
        "task": faulted.task,
        "fault": faulted.fault,
        "seed": faulted.seed,
        "verdict": verdict.value,
        "success": faulted.success,
        "termination": faulted.termination,
        "max_signature_repeats": repeats,
        "steps": faulted.steps,
        "baseline_steps": baseline.steps,
        "extra_steps": extra_steps,
        "tokens": faulted_tokens,
        "baseline_tokens": baseline_tokens,
        "extra_tokens": faulted_tokens - baseline_tokens,
    }
