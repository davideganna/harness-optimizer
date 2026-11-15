"""Structured trace events. Every LLM call (and later every agent step) emits one."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Protocol


class Tracer(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...


class NullTracer:
    def emit(self, event: dict[str, Any]) -> None:
        pass


class MemoryTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append({"ts": time.time(), **event})


class JsonlTracer:
    """Appends one JSON object per line. Cheap, greppable, easy to load into pandas."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps({"ts": time.time(), **event}, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
