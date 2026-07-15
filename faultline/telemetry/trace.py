"""Run trace capture and JSON persistence.

A :class:`RunTrace` is the complete, reproducible record of one agent run:
every tool call (with signature), every model step, token cost, and how the
run terminated. Verdicts are derived from traces alone — no LLM judge.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faultline.telemetry.signatures import signature

#: How a run ended.
TERMINATIONS = ("completed", "gave_up", "max_steps", "crashed")


@dataclass
class RunTrace:
    run_id: str
    model: str
    task: str
    fault: dict[str, Any]  # FaultSpec.to_dict()
    seed: int
    events: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    faults_injected: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    termination: str = "completed"
    success: bool = False
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def is_baseline(self) -> bool:
        return self.fault.get("mode") == "none"

    def signature_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            if e.get("type") == "tool_call":
                s = e["signature"]
                counts[s] = counts.get(s, 0) + 1
        return counts

    def max_signature_repeats(self) -> int:
        counts = self.signature_counts()
        return max(counts.values()) if counts else 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunTrace":
        return cls(**d)

    def save(self, runs_dir: str | Path) -> Path:
        path = Path(runs_dir) / f"{self.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunTrace":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class TraceRecorder:
    """Accumulates events during a run and finalizes them into a RunTrace.

    Its :meth:`on_tool_call` method plugs directly into
    :class:`~faultline.harness.wrapper.ToolWrapper` as the ``on_call`` hook.
    """

    def __init__(
        self,
        model: str,
        task: str,
        fault: dict[str, Any],
        seed: int,
        run_id: str | None = None,
    ):
        self.trace = RunTrace(
            run_id=run_id or f"{task}--{fault.get('mode', 'none')}"
            f"--{fault.get('variant', 'na')}--{model}--s{seed}"
            f"--{uuid.uuid4().hex[:8]}",
            model=model,
            task=task,
            fault=fault,
            seed=seed,
            started_at=time.time(),
        )

    def on_tool_call(self, event: dict[str, Any]) -> None:
        sig = signature(event["tool"], event.get("args") or {})
        self.trace.events.append(
            {
                "type": "tool_call",
                "index": len(self.trace.events),
                "signature": sig,
                **event,
            }
        )
        self.trace.tool_calls += 1
        if event.get("fault_injected"):
            self.trace.faults_injected += 1

    def on_model_step(
        self, tokens_in: int = 0, tokens_out: int = 0, note: str | None = None
    ) -> None:
        self.trace.events.append(
            {
                "type": "model_step",
                "index": len(self.trace.events),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "note": note,
            }
        )
        self.trace.steps += 1
        self.trace.tokens_in += tokens_in
        self.trace.tokens_out += tokens_out

    def finish(
        self,
        termination: str,
        success: bool,
        error: str | None = None,
    ) -> RunTrace:
        assert termination in TERMINATIONS, termination
        self.trace.termination = termination
        self.trace.success = success
        self.trace.error = error
        self.trace.finished_at = time.time()
        return self.trace
