"""The faultline command-line interface."""

from __future__ import annotations

import argparse
import json

from faultline.harness.faults import FaultMode, FaultSpec, FaultVariant


def _cmd_run(args: argparse.Namespace) -> int:
    from faultline.harness.runner import run_one

    fault = (
        FaultSpec.none()
        if args.fault == "none"
        else FaultSpec(
            mode=FaultMode(args.fault),
            variant=FaultVariant(args.variant),
            target_tool=args.target_tool,
            at_call=args.at_call,
        )
    )
    trace = run_one(
        args.task, model=args.model, fault=fault, seed=args.seed,
        runs_dir=args.runs_dir,
    )
    print(
        json.dumps(
            {
                "run_id": trace.run_id,
                "termination": trace.termination,
                "success": trace.success,
                "steps": trace.steps,
                "tool_calls": trace.tool_calls,
                "faults_injected": trace.faults_injected,
                "tokens": trace.tokens_in + trace.tokens_out,
            },
            indent=2,
        )
    )
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from faultline.sweep import run_sweep

    traces = run_sweep(args.config)
    print(f"sweep complete: {len(traces)} runs")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from faultline.analysis.report import generate_report

    out = generate_report(runs_dir=args.runs_dir, out_dir=args.out)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="faultline",
        description="A fault-injection benchmark for LLM agent loops",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one task with one fault")
    p_run.add_argument("--task", required=True, help="path to a task yaml")
    p_run.add_argument("--model", required=True,
                       help="model id, or 'scripted' / 'scripted-stubborn'")
    p_run.add_argument("--fault", default="none",
                       choices=["none"] + [m.value for m in FaultMode
                                           if m is not FaultMode.NONE])
    p_run.add_argument("--variant", default="caught",
                       choices=[v.value for v in FaultVariant])
    p_run.add_argument("--target-tool", default="*")
    p_run.add_argument("--at-call", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--runs-dir", default="runs")
    p_run.set_defaults(fn=_cmd_run)

    p_sweep = sub.add_parser("sweep", help="run a full sweep from a config")
    p_sweep.add_argument("--config", required=True)
    p_sweep.set_defaults(fn=_cmd_sweep)

    p_report = sub.add_parser(
        "report", help="aggregate runs into leaderboard data"
    )
    p_report.add_argument("--runs-dir", default="runs")
    p_report.add_argument("--out", default="site/data")
    p_report.set_defaults(fn=_cmd_report)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
