"""aortacfd-agent — LLM-agent layer over the AortaCFD CFD pipeline.

This package provides a multi-agent supervisor that turns a free-text
clinical referral into a reproducible, literature-grounded patient-specific
CFD simulation with a natural-language clinical summary.

The CFD core itself lives in ``AortaCFD-app`` (included as a git submodule
at ``external/aortacfd-app``). This package never modifies it.

Top-level public API
--------------------

* :class:`~aortacfd_agent.coordinator.Coordinator` — the five-agent supervisor
* :class:`~aortacfd_agent.backends.base.LLMBackend` — provider-agnostic protocol
* :func:`~aortacfd_agent.backends.factory.resolve_backend` — one-line provider switch

Each specialist agent lives under :mod:`aortacfd_agent.agents`. See the
repository README for the architecture diagram and rollout plan.
"""

__version__ = "0.1.0"
