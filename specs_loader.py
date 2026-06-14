"""
specs_loader.py
===============
Loads the markdown agent specs in `specs/` and makes them the single source of
truth for agent behavior. Editing a `.md` changes how the agents think — no code
change needed.

Usage:
    from specs_loader import load_spec, backstory_for
    backstory = backstory_for("researcher")   # role md + shared desk knowledge
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SPECS_DIR = Path(__file__).parent / "specs"

# Knowledge every agent shares (always prepended to a role's backstory).
SHARED = ["00_overview.md", "market_phases.md", "signals.md", "assets.md"]


@lru_cache(maxsize=None)
def load_spec(filename: str) -> str:
    """Read one spec file. Returns '' if missing (so a typo can't crash a run)."""
    path = SPECS_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def shared_context() -> str:
    """Concatenated desk-wide knowledge (overview + phases + signals + assets)."""
    parts = [load_spec(f) for f in SHARED]
    return "\n\n---\n\n".join(p for p in parts if p)


def backstory_for(role_file: str) -> str:
    """
    Full backstory for an agent = its role spec + the shared desk knowledge.

    role_file: 'researcher' | 'analyst' | 'risk_officer' (with or without .md).
    """
    if not role_file.endswith(".md"):
        role_file += ".md"
    role = load_spec(role_file)
    return f"{role}\n\n---\n\n# Shared desk knowledge\n\n{shared_context()}"
