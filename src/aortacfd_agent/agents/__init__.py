"""Five specialist agents, each wrapping a narrow responsibility.

* :mod:`~aortacfd_agent.agents.intake` — clinical text → structured profile
* :mod:`~aortacfd_agent.agents.literature` — RAG corpus → cited parameter choices
* :mod:`~aortacfd_agent.agents.config` — profile + citations → validated config.json
* :mod:`~aortacfd_agent.agents.execution` — run the CFD pipeline via subprocess
* :mod:`~aortacfd_agent.agents.results` — natural-language summary + Q&A

The supervisor that chains these agents together lives in
:mod:`aortacfd_agent.coordinator`.
"""

from .intake import IntakeAgent, IntakeResult

__all__ = ["IntakeAgent", "IntakeResult"]
