import pytest

from faultline.tasks import (
    list_tasks,
    load_task,
    make_environment,
    run_checks,
)


def execute_reference_plan(task, env):
    """Run a task's reference plan directly against the environment."""
    tools = env.tools()
    answer = ""
    for step in task.reference_plan:
        if "answer" in step:
            answer = step["answer"]
        else:
            tools[step["tool"]].fn(**step["args"])
    return answer


def test_all_task_files_load_and_reference_environments():
    paths = list_tasks("tasks")
    assert len(paths) >= 3
    for p in paths:
        task = load_task(p)
        env = make_environment(task.environment)
        tool_names = set(env.tools())
        for step in task.reference_plan:
            if "tool" in step:
                assert step["tool"] in tool_names, (task.name, step["tool"])


@pytest.mark.parametrize(
    "path", [str(p) for p in list_tasks("tasks")], ids=lambda p: p
)
def test_reference_plan_passes_own_checks(path):
    task = load_task(path)
    env = make_environment(task.environment, seed=0)
    answer = execute_reference_plan(task, env)
    result = run_checks(task, env, answer)
    assert result["success"], result["checks"]


def test_checks_fail_on_untouched_environment():
    task = load_task("tasks/flight_rebooking.yaml")
    env = make_environment(task.environment)
    result = run_checks(task, env, "I could not complete the booking.")
    assert not result["success"]


def test_environment_determinism_across_instances():
    a = make_environment("travel", seed=0)
    b = make_environment("travel", seed=0)
    assert a.state == b.state
    a.tools()["book_flight"].fn(flight_id="FL100", passenger="X")
    assert a.state != b.state  # state is per-instance, not shared


def test_travel_env_sold_out_flight_raises():
    env = make_environment("travel")
    with pytest.raises(ValueError, match="sold out"):
        env.tools()["book_flight"].fn(flight_id="FL200", passenger="X")


def test_orders_env_wrong_refund_amount_rejected():
    env = make_environment("orders")
    with pytest.raises(ValueError, match="does not match"):
        env.tools()["refund_order"].fn(order_id="O-1001", amount=5)


def test_unknown_environment_raises():
    with pytest.raises(KeyError):
        make_environment("casino")
