"""Run orchestration.

One run = one (task, model, fault, seed) tuple: build the environment, wrap
its tools with the fault spec, drive the agent, check success
deterministically, and persist the trace.
"""

from __future__ import annotations

from pathlib import Path

from faultline.harness.agents import make_agent
from faultline.harness.faults import FaultMode, FaultSpec, FaultVariant, ToolFaultError
from faultline.harness.wrapper import ToolWrapper
from faultline.tasks.checkers import run_checks
from faultline.tasks.environments import make_environment
from faultline.tasks.loader import TaskSpec, load_task
from faultline.telemetry.trace import RunTrace, TraceRecorder


def run_one(
    task: TaskSpec | str | Path,
    model: str,
    fault: FaultSpec,
    seed: int = 0,
    runs_dir: str | Path = "runs",
) -> RunTrace:
    if not isinstance(task, TaskSpec):
        task = load_task(task)

    env = make_environment(task.environment, seed=seed)
    recorder = TraceRecorder(
        model=model, task=task.name, fault=fault.to_dict(), seed=seed
    )
    wrapper = ToolWrapper(
        tools=env.tools(), fault=fault, on_call=recorder.on_tool_call
    )
    agent = make_agent(model)

    final_answer = ""
    try:
        outcome = agent.run_task(task, wrapper, recorder)
        termination, final_answer = outcome.termination, outcome.final_answer
        error = None
    except ToolFaultError as exc:
        # Propagated variant: the framework never gave the model a chance.
        termination, error = "crashed", f"ToolFaultError: {exc}"
    except Exception as exc:  # loop bug or API failure — also a crash
        termination, error = "crashed", f"{type(exc).__name__}: {exc}"

    check = run_checks(task, env, final_answer)
    trace = recorder.finish(
        termination=termination, success=check["success"], error=error
    )
    trace.events.append(
        {"type": "checks", "index": len(trace.events), **check}
    )
    trace.save(runs_dir)
    return trace


def fault_grid(
    modes: list[str], variants: list[str]
) -> list[FaultSpec]:
    """Expand mode × variant, skipping variants that don't apply.

    EMPTY and MALFORMED are value-shaped, so they run once (as caught);
    the exception-shaped modes run in both variants.
    """
    from faultline.harness.faults import variant_applies

    specs: list[FaultSpec] = []
    for mode_name in modes:
        mode = FaultMode(mode_name)
        if variant_applies(mode):
            for v in variants:
                specs.append(FaultSpec(mode=mode, variant=FaultVariant(v)))
        else:
            specs.append(FaultSpec(mode=mode, variant=FaultVariant.CAUGHT))
    return specs
