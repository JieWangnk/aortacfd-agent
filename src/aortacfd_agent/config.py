"""Shared configuration constants for the agent + paper digest modules."""

from __future__ import annotations

# Default Claude model used across the repo (agent backends + paper digest).
# Haiku 4.5 is plenty for bounded tasks: intake extraction, literature
# retrieval-augmented reasoning, config patching, and abstract-level
# classification in the weekly digest. The Streamlit demo UI can override
# this per-session when a user wants Sonnet for a harder case.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
