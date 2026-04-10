"""JSONL audit trace for a full agent-pipeline run.

Every stage of the coordinator writes one or more entries to a JSONL
file. Each entry records what stage produced it, when, how long it
took, and a compact payload (the input + output of that stage). The
trace is the single source of truth for the "trustworthy AI" provenance
story: given a trace file, a reviewer can reproduce exactly which
inputs flowed into which agent and what each agent decided.

The logger is intentionally minimal — no third-party dependencies, no
rotation, no locking. A single agent run never produces more than a
few kilobytes of trace, so a plain append-mode file handle is fine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """One entry in the agent-run audit trace."""

    stage: str
    timestamp: float
    duration_s: float = 0.0
    status: str = "ok"  # "ok" | "warning" | "error"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 4),
            "status": self.status,
            "payload": self.payload,
        }


class AgentTraceLogger:
    """Append-only JSONL trace logger.

    Parameters
    ----------
    path
        Destination file. Parent directories are created on first write.
    echo
        If True, also log each record to the standard Python logger at
        INFO level. Handy during development.
    """

    def __init__(self, path: Union[str, Path], echo: bool = False):
        self.path = Path(path)
        self.echo = echo
        self._records: List[TraceRecord] = []
        self._open()

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any existing file — each agent run has its own trace.
        self.path.write_text("", encoding="utf-8")

    # -- recording API -------------------------------------------------------

    def record(
        self,
        stage: str,
        payload: Optional[Dict[str, Any]] = None,
        duration_s: float = 0.0,
        status: str = "ok",
    ) -> TraceRecord:
        rec = TraceRecord(
            stage=stage,
            timestamp=time.time(),
            duration_s=duration_s,
            status=status,
            payload=dict(payload or {}),
        )
        self._records.append(rec)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
        if self.echo:
            logger.info("trace %s (%s) in %.2fs", stage, status, duration_s)
        return rec

    def start(self, stage: str) -> "TraceTimer":
        """Return a context manager that records duration automatically."""
        return TraceTimer(self, stage)

    # -- inspection for tests ------------------------------------------------

    @property
    def records(self) -> List[TraceRecord]:
        return list(self._records)

    def stages(self) -> List[str]:
        return [r.stage for r in self._records]


class TraceTimer:
    """Context manager that times a stage and records one entry on exit."""

    def __init__(self, tracer: AgentTraceLogger, stage: str):
        self.tracer = tracer
        self.stage = stage
        self.payload: Dict[str, Any] = {}
        self.status: str = "ok"
        self._start: float = 0.0

    def __enter__(self) -> "TraceTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = time.perf_counter() - self._start
        if exc is not None:
            self.status = "error"
            self.payload.setdefault("exception", f"{exc_type.__name__}: {exc}")
        self.tracer.record(
            stage=self.stage,
            payload=self.payload,
            duration_s=duration,
            status=self.status,
        )
        # Do not suppress exceptions.
        return None

    def set(self, key: str, value: Any) -> None:
        self.payload[key] = value

    def update(self, items: Dict[str, Any]) -> None:
        self.payload.update(items)

    def mark_warning(self) -> None:
        self.status = "warning"
