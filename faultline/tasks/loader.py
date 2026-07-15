"""Task definition loading.

A task YAML declares the prompt, the environment it runs in, a step budget,
deterministic success checks, and a reference plan (the ideal tool-call
sequence, used by the scripted agent and as documentation of intent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskSpec:
    name: str
    description: str
    prompt: str
    environment: str
    max_steps: int
    checks: list[dict[str, Any]]
    reference_plan: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskSpec":
        required = {"name", "prompt", "environment", "checks"}
        missing = required - d.keys()
        if missing:
            raise ValueError(f"task missing fields: {sorted(missing)}")
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            prompt=d["prompt"],
            environment=d["environment"],
            max_steps=int(d.get("max_steps", 15)),
            checks=d["checks"],
            reference_plan=d.get("reference_plan", []),
        )


def load_task(path: str | Path) -> TaskSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TaskSpec.from_dict(data)


def list_tasks(tasks_dir: str | Path = "tasks") -> list[Path]:
    return sorted(Path(tasks_dir).glob("*.yaml"))
