"""Deterministic success checkers.

Checks assert on final environment state and the agent's final answer.
No LLM judging: a task either passes all its checks or it does not, and the
result is reproducible from the trace + environment snapshot.

Check kinds:

- ``state_equals``:    value at ``path`` == ``value``
- ``state_contains``:  ``value`` is a substring/member of the value at ``path``
- ``state_len_gte``:   len(value at ``path``) >= ``value``
- ``answer_contains``: ``value`` is a case-insensitive substring of the
                       final answer

``path`` is a list of keys walked through the state dict (list form, so
keys like file paths may contain dots).
"""

from __future__ import annotations

from typing import Any

from faultline.tasks.environments import Environment
from faultline.tasks.loader import TaskSpec


def _walk(state: dict[str, Any], path: list[Any]) -> Any:
    node: Any = state
    for key in path:
        if isinstance(node, dict):
            node = node[key]
        elif isinstance(node, list):
            node = node[int(key)]
        else:
            raise KeyError(f"cannot descend into {type(node).__name__}")
    return node


def _run_one(
    check: dict[str, Any], env: Environment, final_answer: str
) -> tuple[bool, str]:
    kind = check["kind"]
    value = check.get("value")
    if kind == "answer_contains":
        ok = str(value).casefold() in (final_answer or "").casefold()
        return ok, f"answer contains {value!r}"
    path = check["path"]
    try:
        actual = _walk(env.state, path)
    except (KeyError, IndexError):
        return False, f"{'.'.join(map(str, path))} missing"
    label = ".".join(map(str, path))
    if kind == "state_equals":
        return actual == value, f"{label} == {value!r} (got {actual!r})"
    if kind == "state_contains":
        try:
            ok = value in actual
        except TypeError:
            ok = False
        return ok, f"{label} contains {value!r}"
    if kind == "state_len_gte":
        try:
            ok = len(actual) >= int(value)
        except TypeError:
            ok = False
        return ok, f"len({label}) >= {value}"
    raise ValueError(f"unknown check kind: {kind}")


def run_checks(
    task: TaskSpec, env: Environment, final_answer: str
) -> dict[str, Any]:
    results = []
    for check in task.checks:
        ok, detail = _run_one(check, env, final_answer)
        results.append({"ok": ok, "kind": check["kind"], "detail": detail})
    return {"success": all(r["ok"] for r in results), "checks": results}
