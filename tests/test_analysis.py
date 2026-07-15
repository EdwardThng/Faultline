import json

import pytest

from faultline.analysis import (
    aggregate_runs,
    generate_report,
    load_runs,
    wilson_interval,
)
from faultline.analysis.stats import median
from faultline.cli import main


def test_wilson_interval_known_value():
    low, high = wilson_interval(8, 10)
    assert 0.49 < low < 0.51
    assert 0.93 < high < 0.95


def test_wilson_interval_bounds_and_edges():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(0, 5)
    assert low == 0.0 and high < 0.5
    low, high = wilson_interval(5, 5)
    assert low > 0.5 and high == 1.0


def test_median():
    assert median([]) is None
    assert median([3]) == 3
    assert median([1, 2, 3, 4]) == 2.5


@pytest.fixture(scope="module")
def sweep_runs(tmp_path_factory):
    """A small real sweep: scripted + stubborn over one task."""
    tmp = tmp_path_factory.mktemp("sweep")
    config = tmp / "sweep.yaml"
    config.write_text(
        "models: [scripted, scripted-stubborn]\n"
        "tasks: tasks/flight_rebooking.yaml\n"
        "seeds: [0, 1]\n"
        "fault_modes: [hard_error, transient, empty]\n"
        "variants: [caught, propagated]\n"
        f"runs_dir: {(tmp / 'runs').as_posix()}\n",
        encoding="utf-8",
    )
    from faultline.sweep import run_sweep

    run_sweep(config, verbose=False)
    return tmp / "runs"


def test_aggregate_matrix_shape_and_rates(sweep_runs):
    board = aggregate_runs(load_runs(sweep_runs))
    assert board["models"] == ["scripted", "scripted-stubborn"]
    # hard_error x2 variants + transient x2 + empty = 5 fault keys
    assert len(board["fault_keys"]) == 5
    cells = {(c["model"], c["fault_key"]): c for c in board["cells"]}

    transient = cells[("scripted", "transient/caught")]
    assert transient["n"] == 2 and transient["recovery_rate"] == 1.0
    assert transient["ci_low"] < 1.0 <= transient["ci_high"]

    hard_prop = cells[("scripted", "hard_error/propagated")]
    assert hard_prop["verdicts"]["crashed"] == 2

    stubborn_hard = cells[("scripted-stubborn", "hard_error/caught")]
    assert stubborn_hard["doom_loop_rate"] == 1.0
    assert stubborn_hard["median_loop_len"] >= 5


def test_caught_vs_propagated_delta_present(sweep_runs):
    board = aggregate_runs(load_runs(sweep_runs))
    deltas = {
        (d["model"], d["mode"]): d for d in board["caught_vs_propagated"]
    }
    d = deltas[("scripted", "transient")]
    # scripted recovers transient when caught, crashes when propagated
    assert d["caught_recovery"] == 1.0
    assert d["propagated_recovery"] == 0.0
    assert d["delta"] == 1.0


def test_every_cell_links_to_inspectable_runs(sweep_runs):
    board = aggregate_runs(load_runs(sweep_runs))
    for cell in board["cells"]:
        assert len(cell["run_ids"]) == cell["n"]
        for run_id in cell["run_ids"]:
            assert run_id in board["runs"]


def test_missing_baseline_is_an_error(tmp_path):
    from faultline.harness.faults import FaultMode, FaultSpec
    from faultline.harness.runner import run_one

    run_one("tasks/order_refund.yaml", model="scripted",
            fault=FaultSpec(mode=FaultMode.EMPTY), runs_dir=tmp_path)
    with pytest.raises(ValueError, match="no baseline"):
        aggregate_runs(load_runs(tmp_path))


def test_report_cli_writes_leaderboard_and_traces(sweep_runs, tmp_path,
                                                  capsys):
    out = tmp_path / "data"
    rc = main(["report", "--runs-dir", str(sweep_runs), "--out", str(out)])
    assert rc == 0
    board = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert board["total_runs"] == len(list(sweep_runs.glob("*.json")))
    assert board["thresholds"]["doom_loop_repeats"] == 5
    traces = list((out / "traces").glob("*.json"))
    assert len(traces) == board["total_runs"]
