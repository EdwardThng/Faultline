import pytest

from faultline.harness import (
    FaultMode,
    FaultSpec,
    FaultVariant,
    Tool,
    ToolFaultError,
    ToolWrapper,
)
from faultline.harness.faults import corrupt, variant_applies


def make_tools():
    return {
        "search": Tool(
            name="search",
            description="look up flights",
            parameters={"query": "str"},
            fn=lambda query: [{"flight": "FL123", "price": 250}],
        ),
        "book": Tool(
            name="book",
            description="book a flight",
            parameters={"flight": "str"},
            fn=lambda flight: {"confirmation": f"CONF-{flight}"},
        ),
    }


def test_baseline_passthrough():
    w = ToolWrapper(tools=make_tools())
    r = w.call("search", {"query": "SFO to JFK"})
    assert r.ok and r.value[0]["flight"] == "FL123"
    assert not r.fault_injected
    assert w.injections == 0


def test_unknown_tool_is_error_not_crash():
    w = ToolWrapper(tools=make_tools())
    r = w.call("teleport", {})
    assert not r.ok and "unknown tool" in r.error


def test_hard_error_caught():
    spec = FaultSpec(mode=FaultMode.HARD_ERROR, target_tool="book")
    w = ToolWrapper(tools=make_tools(), fault=spec)
    assert w.call("search", {"query": "x"}).ok  # untargeted tool unaffected
    r = w.call("book", {"flight": "FL123"})
    assert not r.ok and r.fault_injected and "unrecoverable" in r.error


def test_hard_error_propagated_raises_through_loop():
    spec = FaultSpec(
        mode=FaultMode.HARD_ERROR,
        variant=FaultVariant.PROPAGATED,
        target_tool="book",
    )
    w = ToolWrapper(tools=make_tools(), fault=spec)
    with pytest.raises(ToolFaultError) as e:
        w.call("book", {"flight": "FL123"})
    assert e.value.mode is FaultMode.HARD_ERROR and e.value.tool == "book"


def test_transient_fails_n_then_succeeds():
    spec = FaultSpec(
        mode=FaultMode.TRANSIENT, target_tool="book", transient_failures=2
    )
    w = ToolWrapper(tools=make_tools(), fault=spec)
    assert not w.call("book", {"flight": "A"}).ok
    assert not w.call("book", {"flight": "A"}).ok
    r = w.call("book", {"flight": "A"})
    assert r.ok and r.value == {"confirmation": "CONF-A"}
    assert w.injections == 2


def test_empty_returns_none_as_ok():
    spec = FaultSpec(mode=FaultMode.EMPTY, target_tool="search")
    w = ToolWrapper(tools=make_tools(), fault=spec)
    r = w.call("search", {"query": "x"})
    assert r.ok and r.value is None and r.fault_injected


def test_malformed_is_valid_but_wrong():
    spec = FaultSpec(mode=FaultMode.MALFORMED, target_tool="book")
    w = ToolWrapper(tools=make_tools(), fault=spec)
    r = w.call("book", {"flight": "FL123"})
    assert r.ok and r.fault_injected
    assert r.value != {"confirmation": "CONF-FL123"}
    assert isinstance(r.value, dict) and "confirmation" in r.value


def test_timeout_reports_budget():
    spec = FaultSpec(
        mode=FaultMode.TIMEOUT, target_tool="search", timeout_seconds=30
    )
    w = ToolWrapper(tools=make_tools(), fault=spec)
    r = w.call("search", {"query": "x"})
    assert not r.ok and "timed out after 30s" in r.error


def test_at_call_delays_injection():
    spec = FaultSpec(mode=FaultMode.HARD_ERROR, target_tool="search", at_call=3)
    w = ToolWrapper(tools=make_tools(), fault=spec)
    assert w.call("search", {"query": "1"}).ok
    assert w.call("search", {"query": "2"}).ok
    assert not w.call("search", {"query": "3"}).ok


def test_on_call_observer_sees_faulted_calls():
    events = []
    spec = FaultSpec(
        mode=FaultMode.HARD_ERROR,
        variant=FaultVariant.PROPAGATED,
        target_tool="book",
    )
    w = ToolWrapper(tools=make_tools(), fault=spec, on_call=events.append)
    w.call("search", {"query": "x"})
    with pytest.raises(ToolFaultError):
        w.call("book", {"flight": "F"})
    assert len(events) == 2
    assert events[1]["fault_injected"] and events[1]["status"] == "error"


def test_variant_axis_only_for_exception_shaped_modes():
    assert variant_applies(FaultMode.HARD_ERROR)
    assert variant_applies(FaultMode.TIMEOUT)
    assert not variant_applies(FaultMode.EMPTY)
    assert not variant_applies(FaultMode.MALFORMED)


def test_corrupt_is_deterministic_and_type_preserving():
    v = {"price": 250, "flight": "FL123"}
    assert corrupt(v) == corrupt(v)
    assert corrupt(v) != v
    assert set(corrupt(v)) == set(v)
    assert corrupt("abc") == "cba"
    assert corrupt([1, 2]) != [1, 2]
