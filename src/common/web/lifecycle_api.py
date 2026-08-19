"""Routes for the whole-dataset operations: create, rename, delete, merge.

The rules all live in ``common.web.lifecycle``; this layer only reads the
request, decides what needs the operator's ``--allow-delete`` consent, and runs
the slow parts off the event loop.

A merge is the one operation that outlives a request -- it re-encodes every
episode of every source -- so it runs as a JOB: the POST returns an id, the
pane polls it, and only one merge runs at a time.
"""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web  # type: ignore[import]

from common.recording.dataset_edit import ReadOnlyDatasetError
from common.web.lifecycle import (
    delete_dataset,
    directory_size,
    is_working_dir,
    merge_compatibility,
    merge_datasets,
    purge,
    read_dataset_meta,
    rename_dataset,
    valid_dataset_name,
)
from common.web.util import in_executor


def _body_name(body: "dict[str, Any]", key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise web.HTTPBadRequest(text=f"missing {key}")
    return value.strip()


def _require_writable(app: web.Application) -> None:
    if not app["allow_delete"]:
        raise web.HTTPForbidden(text="deletion disabled; restart with --allow-delete")


async def handle_check_name(request: web.Request) -> web.Response:
    """Is this a usable name for a new dataset? Creation itself needs a session.

    A dataset is created by recording into it, so the console cannot make an
    empty one -- what it can do is tell the operator, while they type, that the
    name is fine and free.
    """
    app = request.app
    body = await request.json()
    name = _body_name(body, "name")
    problem = valid_dataset_name(name)
    taken = problem is None and (Path(app["root"]) / name).exists()
    return web.json_response(
        {
            "name": name,
            "ok": problem is None and not taken,
            "problem": problem or (f"{name!r} already exists" if taken else None),
            "exists": taken,
        }
    )


async def handle_rename(request: web.Request) -> web.Response:
    app = request.app
    name = request.match_info["name"]
    body = await request.json()
    new = _body_name(body, "new")
    if is_working_dir(name):
        raise web.HTTPNotFound(text=f"no dataset {name!r}")
    try:
        path = await in_executor(app, rename_dataset, app["root"], name, new)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc))
    except FileExistsError as exc:
        raise web.HTTPConflict(text=str(exc))
    except (ReadOnlyDatasetError, OSError) as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"name": path.name})


async def handle_remove(request: web.Request) -> web.Response:
    """Delete a whole dataset: rename it into the trash now, free the bytes after."""
    app = request.app
    _require_writable(app)
    name = request.match_info["name"]
    if is_working_dir(name):
        raise web.HTTPNotFound(text=f"no dataset {name!r}")
    try:
        trash = await in_executor(app, delete_dataset, app["root"], name)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc))
    except (ReadOnlyDatasetError, OSError) as exc:
        raise web.HTTPConflict(text=str(exc))
    # The dataset is already unreachable under its name; removing the files is
    # slow and nothing waits for it.
    asyncio.get_running_loop().run_in_executor(app["job_executor"], purge, trash)
    return web.json_response({"deleted": name})


def _merge_sources(root: Path, names: "list[str]") -> "list[dict[str, Any]]":
    """What ``merge_compatibility`` needs to judge each source. Blocking."""
    metas = []
    for name in names:
        path = root / name
        if not path.is_dir():
            raise FileNotFoundError(f"no dataset {name!r}")
        meta = read_dataset_meta(path)
        meta["size_bytes"] = directory_size(path)
        metas.append(meta)
    return metas


async def handle_merge_check(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json()
    names = [str(n) for n in body.get("names") or []]
    out_name = _body_name(body, "name")
    reasons = await in_executor(app, _merge_reasons, app["root"], names, out_name)
    return web.json_response({"ok": not reasons, "reasons": reasons})


def _merge_reasons(root: Path, names: "list[str]", out_name: str) -> "list[str]":
    """Blocking: read every source, then apply the pure rules."""
    root = Path(root)
    try:
        metas = _merge_sources(root, names)
    except FileNotFoundError as exc:
        return [str(exc)]
    taken = tuple(p.name for p in root.iterdir() if p.is_dir())
    free = shutil.disk_usage(root).free
    return merge_compatibility(metas, out_name, free, taken)


async def handle_merge(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json()
    names = [str(n) for n in body.get("names") or []]
    out_name = _body_name(body, "name")

    running = [j for j in app["jobs"].values() if j["state"] == "running"]
    if running:
        raise web.HTTPConflict(text=f"a merge is already running ({running[0]['id']})")

    reasons = await in_executor(app, _merge_reasons, app["root"], names, out_name)
    if reasons:
        raise web.HTTPBadRequest(text="; ".join(reasons))

    job = {
        "id": uuid.uuid4().hex[:8],
        "kind": "merge",
        "state": "running",
        "message": "starting",
        "name": out_name,
        "sources": names,
        "started": time.time(),
    }
    app["jobs"][job["id"]] = job

    def run() -> None:
        try:
            merge_datasets(
                Path(app["root"]),
                names,
                out_name,
                progress=lambda m: job.__setitem__("message", m),
            )
            job["state"] = "done"
            job["message"] = f"merged into {out_name}"
        except BaseException as exc:  # noqa: B036 - reported, not swallowed
            job["state"] = "failed"
            job["message"] = str(exc) or exc.__class__.__name__
        finally:
            job["finished"] = time.time()

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response(job)


async def handle_job(request: web.Request) -> web.Response:
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        raise web.HTTPNotFound(text="no such job")
    return web.json_response(job)


def add_lifecycle_routes(app: web.Application) -> None:
    """Register create/rename/delete/merge. ``jobs`` and ``job_executor`` must exist."""
    app.add_routes(
        [
            web.post("/api/datasets/check-name", handle_check_name),
            web.post("/api/datasets/merge", handle_merge),
            web.post("/api/datasets/merge-check", handle_merge_check),
            web.get("/api/datasets/jobs/{id}", handle_job),
            web.post("/api/datasets/{name}/rename", handle_rename),
            web.post("/api/datasets/{name}/remove", handle_remove),
        ]
    )
