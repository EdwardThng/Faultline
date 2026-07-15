"""Tool-call signature hashing.

Signatures hash tool name + normalized arguments, so retries with trivially
different phrasing (case, whitespace, key order) still count as repetition
of the same call.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WS = re.compile(r"\s+")


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return _WS.sub(" ", value).strip().casefold()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k).casefold(): normalize(v) for k, v in value.items()}
    return value


def signature(tool: str, args: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"tool": tool.casefold(), "args": normalize(args)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
