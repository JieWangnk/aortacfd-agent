"""Five specialist agents, each wrapping a narrow responsibility.

* :mod:`~aortacfd_agent.agents.intake` — clinical text → structured profile
* :mod:`~aortacfd_agent.agents.literature` — RAG corpus → cited parameter choices
* :mod:`~aortacfd_agent.agents.config` — profile + citations → validated config.json
* :mod:`~aortacfd_agent.agents.execution` — run the CFD pipeline via subprocess
* :mod:`~aortacfd_agent.agents.results` — natural-language summary + Q&A

The supervisor that chains these agents together lives in
:mod:`aortacfd_agent.coordinator`.
"""

from .config import ConfigAgent, ConfigAgentError, ConfigAgentResult
from .execution import ExecutionAgent, ExecutionAgentError, ExecutionResult
from .intake import IntakeAgent, IntakeResult
from .literature import LiteratureAgent, LiteratureResult

__all__ = [
    "IntakeAgent",
    "IntakeResult",
    "LiteratureAgent",
    "LiteratureResult",
    "ConfigAgent",
    "ConfigAgentError",
    "ConfigAgentResult",
    "ExecutionAgent",
    "ExecutionAgentError",
    "ExecutionResult",
]
