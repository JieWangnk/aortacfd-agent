"""LLM backend adapters.

Every concrete backend (FakeBackend, AnthropicBackend, OpenAICompatBackend)
implements :class:`~aortacfd_agent.backends.base.LLMBackend`. All SDK imports
are lazy — ``pip install anthropic`` is only needed if you actually pick the
Anthropic provider.
"""
