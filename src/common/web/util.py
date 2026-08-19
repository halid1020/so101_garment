"""Small helpers shared by the console's request handlers."""

from __future__ import annotations

import asyncio
from functools import partial

from aiohttp import web  # type: ignore[import]


async def in_executor(app: web.Application, fn, *args):
    """Run a blocking call on the app's worker pool, off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app["executor"], partial(fn, *args))


def preinit_tqdm_lock() -> None:
    """Create tqdm's class lock up front, to stop concurrent loads racing on it.

    LeRobot's dataset loading reaches HuggingFace ``datasets``, whose
    ``thread_map`` wraps work in tqdm's ``ensure_lock``. That helper deletes
    ``tqdm._lock`` again if it was absent when it started, so two loads running
    together in a thread pool both find it absent, both try to delete it, and the
    second raises ``AttributeError: type object 'tqdm' has no attribute '_lock'``
    -- which surfaced as a 500 on an otherwise fine dataset. Creating the lock
    once means it is never absent, so it is never deleted.
    """
    try:
        from tqdm import tqdm

        tqdm.set_lock(tqdm.get_lock())
    except Exception:
        pass
