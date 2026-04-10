"""Tool implementations that agents can call through the LLM tool-use API.

Each module exposes pure-ish Python functions plus a ``ToolSpec`` factory
that registers them with the agent loop. Tools never raise — they return
``{"error": ...}`` dicts so the model can see failures and retry.
"""
