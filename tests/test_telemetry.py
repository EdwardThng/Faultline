from faultline.harness import FaultMode, FaultSpec, Tool, ToolWrapper
from faultline.telemetry import (
    RunTrace,
    TraceRecorder,
    Verdict,
    VerdictThresholds,
    classify,
    signature,
)


# --- signatures -----------------------------------------------------------

def test_signature_ignores_trivial_phrasing_differences():
    a = signature("Search", {"query": "SFO  to JFK "})
    b = signature("search", {"Query": "sfo to jfk"})
    assert a == b


def test_signature_distinguishes_real_differences():
    assert signature("search", {"query": "SFO to JFK"}) != signature(
        "search", {"query": "SFO to LAX"}
    )
    assert signature("search", {"q": "x"}) != signature("book", {"q": "x"})


def test_signature_normalizes_nested_and_numeric():
    a = signature("t", {"filters": {"Max Price": 250.0}})
    b = signature("t", {"filters": {"max price": 250}})
    assert a == b


# --- trace recording ------------------------------------------------------

def make_recorder(mode="none", **fault_kw):
    fault = (
        FaultSpec.none() if mode == "none"
        else FaultSpec(mode=FaultMode(mode), **fault_kw)
    )
    return TraceRecorder(
        model="test-model", task="demo", fault=fault.to_dict(), seed=0
    )


def test_recorder_plugs_into_wrapper_and_counts():
    rec = make_recorder("hard_error", target_tool="book")
    tools = {
        "book": Tool("book", "d", {}, lambda **kw: {"ok": True}),
        "search": Tool("search", "d", {}, lambda **kw: []),
    }
    w = ToolWrapper(
        tools=tools,
        fault=FaultSpec(mode=FaultMode.HARD_ERROR, target_tool="book"),
        on_call=rec.on_tool_call,
    )
    rec.on_model_step(tokens_in=100, tokens_out=20)
    w.call("search", {"q": "x"})
    rec.on_model_step(tokens_in=150, tokens_out=30)
    w.call("book", {"flight": "F1"})
    trace = rec.finish(termination="gave_up", success=False)

    assert trace.steps == 2 and trace.tool_calls == 2
    assert trace.faults_injected == 1
    assert trace.tokens_in == 250 and trace.tokens_out == 50
    assert all("signature" in e for e in trace.events if e["type"] == "tool_call")


def test_trace_save_load_roundtrip(tmp_path):
    rec = make_recorder()
    rec.on_model_step(tokens_in=10, tokens_out=5)
    rec.on_tool_call(
        {"tool": "search", "args": {"q": "x"}, "status": "ok",
         "value": [1], "error": None, "fault_injected": False,
         "latency_ms": 1.0}
    )
    trace = rec.finish(termination="completed", success=True)
    path = trace.save(tmp_path)
    loaded = RunTrace.load(path)
    assert loaded.to_dict() == trace.to_dict()
    assert loaded.is_baseline


def test_max_signature_repeats():
    rec = make_recorder()
    for _ in range(4):
        rec.on_tool_call(
            {"tool": "book", "args": {"flight": " F1 "}, "status": "error",
             "value": None, "error": "boom", "fault_injected": True,
             "latency_ms": 1.0}
        )
    rec.on_tool_call(
        {"tool": "search", "args": {}, "status": "ok", "value": [],
         "error": None, "fault_injected": False, "latency_ms": 1.0}
    )
    assert rec.trace.max_signature_repeats() == 4


# --- verdicts -------------------------------------------------------------

def trace_with(steps, success, termination="completed", repeats=0,
               tokens=0, task="demo", fault_mode="hard_error"):
    t = RunTrace(
        run_id=f"r-{steps}-{success}-{repeats}",
        model="m", task=task,
        fault={"mode": fault_mode, "variant": "caught"},
        seed=0, steps=steps, termination=termination, success=success,
        tokens_in=tokens, tokens_out=0,
    )
    for i in range(repeats):
        t.events.append(
            {"type": "tool_call", "index": i, "signature": "aaa",
             "tool": "book", "args": {}}
        )
    return t


BASELINE = trace_with(steps=6, success=True, fault_mode="none", tokens=1000)


def test_verdict_recovered():
    r = classify(trace_with(steps=8, success=True, tokens=1500), BASELINE)
    assert r["verdict"] == Verdict.RECOVERED.value
    assert r["extra_steps"] == 2


def test_verdict_degraded_by_step_overhead():
    r = classify(trace_with(steps=40, success=True, tokens=1500), BASELINE)
    assert r["verdict"] == Verdict.DEGRADED.value


def test_verdict_degraded_by_token_overhead():
    r = classify(trace_with(steps=7, success=True, tokens=9000), BASELINE)
    assert r["verdict"] == Verdict.DEGRADED.value


def test_verdict_doom_looped():
    r = classify(
        trace_with(steps=20, success=False, termination="max_steps",
                   repeats=7),
        BASELINE,
    )
    assert r["verdict"] == Verdict.DOOM_LOOPED.value
    assert r["max_signature_repeats"] == 7


def test_verdict_spiraled():
    r = classify(
        trace_with(steps=15, success=False, termination="gave_up", repeats=2),
        BASELINE,
    )
    assert r["verdict"] == Verdict.SPIRALED.value


def test_verdict_crashed_wins_over_everything():
    r = classify(
        trace_with(steps=3, success=False, termination="crashed", repeats=9),
        BASELINE,
    )
    assert r["verdict"] == Verdict.CRASHED.value


def test_sensible_retries_are_not_doom_loops():
    # transient fault: 2 retries then success, modest overhead
    r = classify(trace_with(steps=9, success=True, repeats=3, tokens=1800),
                 BASELINE)
    assert r["verdict"] == Verdict.RECOVERED.value


def test_thresholds_load_from_repo_config():
    t = VerdictThresholds.from_config("analysis/config.yaml")
    assert t.doom_loop_repeats == 5
    assert t.recovered_max_step_overhead_ratio == 1.5
