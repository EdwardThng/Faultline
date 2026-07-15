"""Agent adapters.

The runner is framework-agnostic: an agent is anything with a ``run_task``
method that drives tool calls through the FaultLine wrapper. Two adapters
ship with the reference implementation:

- ScriptedAgent: deterministic, follows the task's reference plan with a
  configurable retry policy. Used for tests and pipeline demos — no API key
  or network required.
- ClaudeAgent: a real model driven through the Anthropic API (requires the
  ``anthropic`` package and credentials).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faultline.harness.wrapper import ToolWrapper
from faultline.tasks.loader import TaskSpec
from faultline.telemetry.trace import TraceRecorder


@dataclass
class AgentOutcome:
    termination: str  # completed | gave_up | max_steps (crashed set by runner)
    final_answer: str


class ScriptedAgent:
    """Follows the task's reference plan; retries failed calls up to
    ``max_retries`` times, then gives up.

    A small ``max_retries`` models a sensible agent; a large one models a
    doom-looper that keeps hammering the same failing call.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def run_task(
        self,
        task: TaskSpec,
        wrapper: ToolWrapper,
        recorder: TraceRecorder,
    ) -> AgentOutcome:
        answer = ""
        steps = 0
        for plan_step in task.reference_plan:
            if "answer" in plan_step:
                answer = plan_step["answer"]
                continue
            attempts = 0
            while True:
                if steps >= task.max_steps:
                    return AgentOutcome("max_steps", answer)
                steps += 1
                # synthetic but deterministic token accounting
                recorder.on_model_step(
                    tokens_in=200 + 60 * steps, tokens_out=40
                )
                result = wrapper.call(plan_step["tool"], plan_step["args"])
                if result.ok:
                    break  # proceeds naively even on empty/malformed values
                attempts += 1
                if attempts > self.max_retries:
                    return AgentOutcome(
                        "gave_up",
                        f"I could not complete the task: "
                        f"{plan_step['tool']} kept failing ({result.error}).",
                    )
        return AgentOutcome("completed", answer)


class ClaudeAgent:
    """Drives a Claude model through a manual tool-use loop.

    The manual loop (rather than the SDK's tool runner) is deliberate:
    every tool call must pass through the FaultLine wrapper, and propagated
    faults must be able to raise straight through the loop to model
    loop-crash failures.
    """

    def __init__(self, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens

    def _tool_defs(self, wrapper: ToolWrapper) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        arg: {"type": "string", "description": desc}
                        for arg, desc in t.parameters.items()
                    },
                    "required": list(t.parameters),
                },
            }
            for t in wrapper.tools.values()
        ]

    def run_task(
        self,
        task: TaskSpec,
        wrapper: ToolWrapper,
        recorder: TraceRecorder,
    ) -> AgentOutcome:
        import anthropic

        client = anthropic.Anthropic()
        system = (
            "You are an agent completing a task with tools. Use the tools "
            "provided, then report the result. If a tool fails, decide "
            "whether to retry, work around it, or report that you could "
            "not finish."
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.prompt}
        ]
        tools = self._tool_defs(wrapper)
        final_text = ""

        for _ in range(task.max_steps):
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
            recorder.on_model_step(
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                note=response.stop_reason,
            )
            final_text = next(
                (b.text for b in response.content if b.type == "text"),
                final_text,
            )
            if response.stop_reason == "pause_turn":
                messages.append(
                    {"role": "assistant", "content": response.content}
                )
                continue
            if response.stop_reason != "tool_use":
                termination = (
                    "completed" if response.stop_reason == "end_turn"
                    else "gave_up"
                )
                return AgentOutcome(termination, final_text)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                # Propagated faults raise ToolFaultError here, through the
                # loop, exactly as they would in an unguarded framework.
                result = wrapper.call(block.name, dict(block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.as_model_text(),
                        "is_error": not result.ok,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return AgentOutcome("max_steps", final_text)


def make_agent(model: str):
    """Resolve a --model string to an agent adapter.

    ``scripted`` / ``scripted-stubborn`` are the deterministic reference
    agents; anything else is treated as a Claude model id.
    """
    if model == "scripted":
        return ScriptedAgent(max_retries=2)
    if model == "scripted-stubborn":
        return ScriptedAgent(max_retries=50)
    return ClaudeAgent(model=model)
