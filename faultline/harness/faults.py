"""Fault specifications and injection behavior.

A fault is described declaratively by a :class:`FaultSpec` and applied at the
tool-call boundary by the wrapper. Faults never touch the agent framework's
internals; they only change what a tool call returns (or raises).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultMode(str, Enum):
    HARD_ERROR = "hard_error"
    TRANSIENT = "transient"
    MALFORMED = "malformed"
    EMPTY = "empty"
    TIMEOUT = "timeout"

    NONE = "none"  # baseline runs


class FaultVariant(str, Enum):
    # Error is surfaced to the model as a tool result.
    CAUGHT = "caught"
    # Exception is raised through the agent loop.
    PROPAGATED = "propagated"


#: Modes whose faults are exception-shaped and can therefore either be
#: surfaced to the model (caught) or raised through the loop (propagated).
#: EMPTY and MALFORMED produce ordinary-looking values, so the variant axis
#: does not apply to them and sweeps run them once.
VARIANT_MODES = frozenset(
    {"hard_error", "transient", "timeout"}
)


def variant_applies(mode: "FaultMode") -> bool:
    return mode.value in VARIANT_MODES


class ToolFaultError(RuntimeError):
    """Raised through the agent loop in the ``propagated`` variant."""

    def __init__(self, mode: FaultMode, tool: str, message: str):
        super().__init__(message)
        self.mode = mode
        self.tool = tool


@dataclass
class FaultSpec:
    """Declarative description of one injected fault.

    target_tool: tool name the fault applies to, or "*" for any tool.
    at_call: 1-based index among calls to the target tool at which the
        fault starts firing.
    transient_failures: for TRANSIENT, how many consecutive calls fail
        before the tool succeeds again.
    timeout_seconds: reported timeout budget (simulated; no real sleep).
    """

    mode: FaultMode
    variant: FaultVariant = FaultVariant.CAUGHT
    target_tool: str = "*"
    at_call: int = 1
    transient_failures: int = 2
    timeout_seconds: float = 30.0

    @classmethod
    def none(cls) -> "FaultSpec":
        return cls(mode=FaultMode.NONE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "variant": self.variant.value,
            "target_tool": self.target_tool,
            "at_call": self.at_call,
            "transient_failures": self.transient_failures,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class FaultInjector:
    """Stateful applicator of a FaultSpec across the calls of one run."""

    spec: FaultSpec
    _seen: dict[str, int] = field(default_factory=dict)
    _transient_fired: int = 0
    injections: int = 0

    def matches(self, tool_name: str) -> bool:
        if self.spec.mode is FaultMode.NONE:
            return False
        return self.spec.target_tool in ("*", tool_name)

    def should_fire(self, tool_name: str) -> bool:
        """Record one call to *tool_name* and decide whether to inject."""
        if not self.matches(tool_name):
            return False
        count = self._seen.get(tool_name, 0) + 1
        self._seen[tool_name] = count
        if count < self.spec.at_call:
            return False
        if self.spec.mode is FaultMode.TRANSIENT:
            if self._transient_fired >= self.spec.transient_failures:
                return False
            self._transient_fired += 1
        self.injections += 1
        return True

    def error_message(self, tool_name: str) -> str:
        mode = self.spec.mode
        if mode is FaultMode.HARD_ERROR:
            return f"{tool_name} failed: internal error (unrecoverable)"
        if mode is FaultMode.TRANSIENT:
            return f"{tool_name} failed: service temporarily unavailable, try again"
        if mode is FaultMode.TIMEOUT:
            return (
                f"{tool_name} timed out after {self.spec.timeout_seconds:g}s"
            )
        return f"{tool_name} fault: {mode.value}"


def corrupt(value: Any) -> Any:
    """Produce a syntactically valid but semantically wrong version of *value*.

    Deterministic, so malformed runs are reproducible.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value * 10 + 7
    if isinstance(value, float):
        return -value if value else 99.9
    if isinstance(value, str):
        return value[::-1] if value else "OK"
    if isinstance(value, list):
        return list(reversed([corrupt(v) for v in value])) if value else [None]
    if isinstance(value, dict):
        out = copy.deepcopy(value)
        keys = sorted(out, key=str)
        vals = [out[k] for k in keys]
        # rotate values across keys so every field is present but wrong
        for k, v in zip(keys, vals[1:] + vals[:1]):
            out[k] = v
        if len(keys) < 2:
            for k in keys:
                out[k] = corrupt(out[k])
        return out
    return value
