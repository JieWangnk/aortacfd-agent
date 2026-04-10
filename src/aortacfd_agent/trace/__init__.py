"""JSONL audit trace for every agent run.

Records one entry per model call and one entry per tool call, with inputs,
outputs, and timing. The trace file is the single source of truth for the
"trustworthy AI" provenance story.
"""
