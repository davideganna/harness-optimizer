"""Response caches keyed by a hash of the full request.

Caching makes re-running experiments cheap and deterministic, which is what makes
`replay` and ablations practical. Use `sample_index` on the request when you *want*
multiple independent samples of the same prompt.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from harness_optimizer.llm.base import LLMResponse


class ResponseCache(Protocol):
    def get(self, key: str) -> LLMResponse | None: ...
    def set(self, key: str, response: LLMResponse) -> None: ...


class NullCache:
    def get(self, key: str) -> LLMResponse | None:
        return None

    def set(self, key: str, response: LLMResponse) -> None:
        pass


class MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> LLMResponse | None:
        raw = self._data.get(key)
        return LLMResponse.model_validate_json(raw) if raw else None

    def set(self, key: str, response: LLMResponse) -> None:
        self._data[key] = response.model_dump_json()


class SQLiteCache:
    def __init__(self, path: str | Path = ".harness_cache/llm.sqlite") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def set(self, key: str, response: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
                (key, response.model_dump_json()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
