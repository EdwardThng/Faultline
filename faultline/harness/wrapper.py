"""The FaultLine tool wrapper.

Wraps tools at the call boundary so faults are injected without touching the
agent framework's internals. Any agent loop that calls tools through
:meth:`ToolWrapper.call` can be tested.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from faultline.harness.faults import (
    FaultInjector,
    FaultMode,
    FaultSpec,
    FaultVariant,
    ToolFaultError,
    corrupt,
)


@dataclass
class Tool:
    """A callable tool plus the metadata an agent needs to invoke it."""

    name: str
    description: str
    parameters: dict[str, str]  # arg name -> human-readable type/description
    fn: Callable[..., Any]


@dataclass
class ToolResult:
    """What the agent loop receives back from a wrapped tool call."""

    status: str  # "ok" | "error"
    value: Any = None
    error: str | None = None
    fault_injected: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_model_text(self) -> str:
        """Render the result the way it would be surfaced to the model."""
        if self.ok:
            return repr(self.value)
        return f"ERROR: {self.error}"


@dataclass
class ToolWrapper:
    """Dispatches agent tool calls to real tools, injecting faults per spec.

    on_call: optional observer invoked with a telemetry event dict after
        every call (including faulted ones that raise).
    """

    tools: dict[str, Tool]
    fault: FaultSpec = field(default_factory=FaultSpec.none)
    on_call: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        self._injector = FaultInjector(self.fault)

    @property
    def injections(self) -> int:
        return self._injector.injections

    def call(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute one tool call, applying the fault spec.

        In the ``propagated`` variant a fault raises :class:`ToolFaultError`
        through the loop; in the ``caught`` variant it comes back as an
        error-status :class:`ToolResult`.
        """
        started = time.perf_counter()
        if tool_name not in self.tools:
            result = ToolResult(
                status="error", error=f"unknown tool: {tool_name}"
            )
            self._emit(tool_name, args, result, started)
            return result

        fire = self._injector.should_fire(tool_name)
        if fire:
            result = self._inject(tool_name, args, started)
            return result  # caught variants only; propagated raised inside

        try:
            value = self.tools[tool_name].fn(**args)
            result = ToolResult(status="ok", value=value)
        except Exception as exc:  # real tool bug, not an injected fault
            result = ToolResult(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            traceback.print_exc()
        self._emit(tool_name, args, result, started)
        return result

    def _inject(
        self, tool_name: str, args: dict[str, Any], started: float
    ) -> ToolResult:
        mode = self.fault.mode

        if mode is FaultMode.EMPTY:
            result = ToolResult(status="ok", value=None, fault_injected=True)
            self._emit(tool_name, args, result, started)
            return result

        if mode is FaultMode.MALFORMED:
            try:
                real = self.tools[tool_name].fn(**args)
            except Exception:
                real = None
            result = ToolResult(
                status="ok", value=corrupt(real), fault_injected=True
            )
            self._emit(tool_name, args, result, started)
            return result

        # hard_error / transient / timeout are error-shaped
        message = self._injector.error_message(tool_name)
        result = ToolResult(status="error", error=message, fault_injected=True)
        self._emit(tool_name, args, result, started)
        if self.fault.variant is FaultVariant.PROPAGATED:
            raise ToolFaultError(mode, tool_name, message)
        return result

    def _emit(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        started: float,
    ) -> None:
        if self.on_call is None:
            return
        self.on_call(
            {
                "tool": tool_name,
                "args": args,
                "status": result.status,
                "value": result.value,
                "error": result.error,
                "fault_injected": result.fault_injected,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
