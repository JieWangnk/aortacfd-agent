"""Stub that masks pydantic inside the CFD subprocess.

Why this exists
---------------

The AortaCFD-app submodule pinned at ``afeffe5a`` has a latent bug in
``src/config/schema.py::PhysicsConfig.sync_simulation_type``: under
pydantic v2, ``use_enum_values = True`` coerces ``self.model`` to a
bare string at validation time, then the after-validator calls
``self.model.value`` on that string and crashes::

    AttributeError: 'str' object has no attribute 'value'

The crash is triggered whenever anyone constructs ``PhysicsConfig``
with a non-default model — so any programmatically generated config
that sets ``physics.model`` fails hard.

``patient_runner/core.py`` only runs this validator when
``is_pydantic_available()`` returns True — otherwise it treats the
config as a plain dict and the downstream code is perfectly happy.
So the cleanest unblock is to make the submodule subprocess *think*
pydantic is not installed, while our agent process (which needs real
pydantic for the anthropic SDK and for jsonschema validation) keeps
using it.

ExecutionAgent prepends this directory to ``PYTHONPATH`` only for the
child process environment, so the mask is scoped exactly to the
subprocess that would have hit the bug.

Remove this shim (and the corresponding PYTHONPATH wiring in
:class:`~aortacfd_agent.agents.execution.ExecutionAgent`) once upstream
AortaCFD-app ships a fix for the ``.value``-on-str bug.
"""

raise ImportError(
    "pydantic is masked for the AortaCFD-app subprocess to force the "
    "dataclass fallback path in config.schema. This is a deliberate "
    "workaround for a latent bug in the pinned submodule commit. See "
    "aortacfd_agent/_subproc_shims/pydantic_mask/pydantic.py for details."
)
