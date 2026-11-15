"""Environment / secrets loading.

API keys are read from the environment. For local development, put them in a `.env` file
at the project root (it is git-ignored); `load_env()` loads it without overriding
variables that are already set in the shell.
"""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_env() -> bool:
    """Load the nearest `.env` (searching upward from the current directory)."""
    return load_dotenv(find_dotenv(usecwd=True), override=False)
