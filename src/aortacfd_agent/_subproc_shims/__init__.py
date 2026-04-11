"""Shim directory — packages placed here are prepended to the subprocess
PYTHONPATH by :class:`aortacfd_agent.agents.execution.ExecutionAgent`.

These shims are only ever loaded by the CFD subprocess, never by the
agent process itself. See ``pydantic_mask/`` for the one shim we ship
today and the reason it exists.
"""
