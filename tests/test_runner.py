import json

from faultline.cli import main
from faultline.harness.faults import FaultMode, FaultSpec, FaultVariant
from faultline.harness.runner import fault_grid, run_one
from faultline.telemetry import Verdict, classify


TASK = "tasks/flight_rebooking.yaml"


def run(model="scripted", mode=None, variant="caught", tmp_path=".", **kw):
    fault = (
        FaultSpec.none() if mode is None
        else FaultSpec(mode=FaultMode(mode), variant=FaultVariant(variant),
                       **kw)
    )
    return run_one(TASK, model=model, fault=fault, seed=0,
                   runs_dir=tmp_path)


def test_baseline_completes_and_saves_trace(tmp_path):
    trace = run(tmp_path=tmp_path)
    assert trace.success and trace.termination == "completed"
    assert trace.faults_injected == 0
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text(encoding="utf-8"))
    assert data["run_id"] == trace.run_id
    assert data["events"][-1]["type"] == "checks"


def test_transient_fault_is_recovered_by_retrying(tmp_path):
    trace = run(mode="transient", tmp_path=tmp_path)
    assert trace.success and trace.termination == "completed"
    assert trace.faults_injected == 2
    baseline = run(tmp_path=tmp_path)
    verdict = classify(trace, baseline)
    assert verdict["verdict"] == Verdict.RECOVERED.value


def test_hard_error_caught_makes_scripted_agent_give_up(tmp_path):
    trace = run(mode="hard_error", tmp_path=tmp_path)
    assert not trace.success and trace.termination == "gave_up"
    baseline = run(tmp_path=tmp_path)
    assert classify(trace, baseline)["verdict"] == Verdict.SPIRALED.value


def test_hard_error_propagated_crashes_the_loop(tmp_path):
    trace = run(mode="hard_error", variant="propagated", tmp_path=tmp_path)
    assert trace.termination == "crashed"
    assert "ToolFaultError" in trace.error
    baseline = run(tmp_path=tmp_path)
    assert classify(trace, baseline)["verdict"] == Verdict.CRASHED.value


def test_stubborn_agent_doom_loops_on_hard_error(tmp_path):
    trace = run(model="scripted-stubborn", mode="hard_error",
                tmp_path=tmp_path)
    assert not trace.success and trace.termination == "max_steps"
    assert trace.max_signature_repeats() >= 5
    baseline = run(model="scripted-stubborn", tmp_path=tmp_path)
    assert classify(trace, baseline)["verdict"] == Verdict.DOOM_LOOPED.value


def test_fault_grid_skips_variants_for_value_shaped_modes():
    specs = fault_grid(
        ["hard_error", "transient", "malformed", "empty", "timeout"],
        ["caught", "propagated"],
    )
    # 3 exception-shaped x 2 variants + 2 value-shaped x 1 = 8
    assert len(specs) == 8


def test_cli_run_end_to_end(tmp_path, capsys):
    rc = main([
        "run", "--task", TASK, "--model", "scripted",
        "--fault", "timeout", "--variant", "caught",
        "--seed", "0", "--runs-dir", str(tmp_path),
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["termination"] == "gave_up" and not out["success"]


def test_cli_sweep_small_config(tmp_path, capsys):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        "models: [scripted]\n"
        "tasks: tasks/order_refund.yaml\n"
        "seeds: [0]\n"
        "fault_modes: [transient, empty]\n"
        "variants: [caught, propagated]\n"
        f"runs_dir: {tmp_path.as_posix()}/runs\n",
        encoding="utf-8",
    )
    rc = main(["sweep", "--config", str(config)])
    assert rc == 0
    # baseline + transient x 2 variants + empty x 1 = 4 runs
    assert len(list((tmp_path / "runs").glob("*.json"))) == 4
