"""Serve a trained checkpoint's action chunks over HTTP (the GPU half of on-robot eval).

The rig's computer holds the cameras and the follower buses; it does not have to
hold the GPU. This runs a checkpoint on whatever machine does, answering the one
question the control loop needs answered: given the last few observations, what
are the next actions?

It is deliberately not a robot controller. It never sees a motor, never decides
when to move, and holds no state between requests: every request calls
``policy.reset()`` and predicts from the observation window it was handed. That
makes a run reproducible, makes a dropped connection harmless, and means a
client that dies mid-episode cannot leave stale history behind to poison the
next one. The client keeps the timing and the safety; this keeps the weights.

Chunking is what makes the split cheap. An action-chunking policy plans many
steps from one observation -- a hundred for the ACT configuration trained here,
thirty-two for diffusion -- so one request covers seconds of motion rather than a
single control tick, and the network sits outside the control loop instead of
inside it.

Two details are load-bearing and are the reason this file exists rather than a
few lines of glue:

  * Policies with more than one observation step (diffusion here) were trained on
    *adjacent* frames. The request therefore carries a whole window, and the
    batch is stacked along a time axis -- LeRobot's own diffusion code asserts
    the window is exactly ``n_obs_steps`` long.
  * The post-processor unnormalises one action at a time, so a chunk is pushed
    through it step by step. Handing it a whole chunk is the obvious thing to try
    and it is wrong; LeRobot's own inference server does the same loop.

Run it (on the GPU box, bound to loopback and reached through an SSH tunnel --
see documents/remote_policy_inference.md):

    venv/bin/python tool/policy_server.py \\
        --checkpoint <run>/train/act/checkpoints/last/pretrained_model

Then, on the rig:

    ssh -N -L 8765:127.0.0.1:8765 <user>@<host>          # in another terminal
    venv/bin/python tool/run_policy.py --server http://127.0.0.1:8765 \\
        --task "..." --dry-run

EXACTLY ONE of --server and --checkpoint, never both: the client takes one
source of actions and refuses two. Which checkpoint answered is reported over
the wire in /meta and written into the run log, so naming it again on the rig
would only be a second chance to name it wrongly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

# This server only ever reads a checkpoint from local disk. Without this, a
# checkpoint directory missing an optional file sends LeRobot to the Hub and the
# failure that surfaces is a 401, which says nothing about the real problem.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from actoris_harena.deploy.policy_wire import (  # noqa: E402
    WireError,
    decode_request,
    encode_chunk,
)
from aiohttp import web  # noqa: E402

_STATE_KEY = "observation.state"
_IMAGE_PREFIX = "observation.images."


class PolicyHost:
    """One loaded checkpoint, plus the batch assembly its inputs require."""

    def __init__(self, checkpoint: str, device: str) -> None:
        import torch

        from tool.eval_sim_policy import load_policy

        self.torch = torch
        self.checkpoint = checkpoint
        self.device = device
        self.policy, self.pre, self.post, self.type = load_policy(checkpoint, device)
        cfg = self.policy.config
        self.n_obs_steps = int(getattr(cfg, "n_obs_steps", 1) or 1)
        self.n_action_steps = int(getattr(cfg, "n_action_steps", 1) or 1)
        self.image_keys = sorted(getattr(cfg, "image_features", {}) or {})
        self.cameras = [k[len(_IMAGE_PREFIX) :] for k in self.image_keys]
        action = (getattr(cfg, "output_features", {}) or {}).get("action")
        self.action_dim = int(action.shape[0]) if action is not None else 12

    def meta(self) -> dict:
        return {
            "policy_type": self.type,
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "action_dim": self.action_dim,
            "cameras": self.cameras,
        }

    def _batch(self, steps, task: str) -> dict:
        """An observation window -> one policy batch.

        Each step goes through the same ``build_batch`` + pre-processor the
        single-observation path uses, so a frame is normalised identically
        whether inference runs here or on the rig; only then are the steps
        stacked along a new time axis. Doing it the other way round -- stacking
        raw frames and normalising once -- would be equivalent in arithmetic and
        far easier to get subtly wrong.
        """
        from tool.eval_sim_policy import build_batch

        torch = self.torch
        per_step = []
        for state, images in steps:
            batch = build_batch(
                np.asarray(state, dtype=np.float32), images, task, self.device
            )
            per_step.append(self.pre(batch))

        if self.n_obs_steps == 1:
            return per_step[0]

        stacked = dict(per_step[-1])
        for key in [_STATE_KEY, *self.image_keys]:
            if key in stacked:
                stacked[key] = torch.stack([b[key] for b in per_step], dim=1)
        return stacked

    def predict(
        self, steps, task: str, actions: "int | None"
    ) -> "tuple[np.ndarray, dict]":
        """Observation window -> ``(K, action_dim)`` actions, and timings."""
        torch = self.torch
        t0 = time.perf_counter()
        self.policy.reset()
        batch = self._batch(steps, task)
        t_prep = time.perf_counter()

        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : self.n_action_steps, :]
            if actions:
                chunk = chunk[:, : int(actions), :]
            # The post-processor unnormalises one action at a time.
            steps_out = [self.post(chunk[:, i, :]) for i in range(chunk.shape[1])]
            out = torch.stack(steps_out, dim=1).squeeze(0)
        t_infer = time.perf_counter()

        arr = np.asarray(out.detach().to("cpu"), dtype=np.float32).reshape(
            -1, self.action_dim
        )
        return arr, {
            "prepare_s": round(t_prep - t0, 4),
            "infer_s": round(t_infer - t_prep, 4),
        }


def build_app(host: PolicyHost) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    lock = asyncio.Lock()
    session = {"id": ""}

    async def meta(_request: web.Request) -> web.Response:
        return web.json_response(host.meta())

    async def reset(_request: web.Request) -> web.Response:
        async with lock:
            session["id"] = uuid.uuid4().hex
            await asyncio.get_running_loop().run_in_executor(None, host.policy.reset)
        print(f"↻ session {session['id'][:8]} started")
        return web.json_response({"session": session["id"], **host.meta()})

    async def act(request: web.Request) -> web.Response:
        body = await request.read()
        try:
            req = decode_request(body)
        except WireError as exc:
            raise web.HTTPBadRequest(text=f"bad request: {exc}")

        if session["id"] and req["session"] and req["session"] != session["id"]:
            # Two clients, or a client that restarted without resetting. Refuse
            # rather than answer: the other one may have arms under torque.
            raise web.HTTPConflict(text="session is not the one that reset this server")
        if sorted(req["cameras"]) != sorted(host.cameras):
            raise web.HTTPBadRequest(
                text=(
                    f"camera set {sorted(req['cameras'])} does not match the "
                    f"checkpoint's {sorted(host.cameras)}"
                )
            )
        if len(req["steps"]) != host.n_obs_steps:
            raise web.HTTPBadRequest(
                text=(
                    f"policy needs {host.n_obs_steps} observation step(s), "
                    f"got {len(req['steps'])}"
                )
            )

        t0 = time.perf_counter()
        async with lock:
            try:
                arr, timings = await asyncio.get_running_loop().run_in_executor(
                    None, host.predict, req["steps"], req["task"], req["actions"]
                )
            except Exception as exc:  # noqa: BLE001 - report, never crash the server
                print(f"❌ inference failed: {type(exc).__name__}: {exc}")
                raise web.HTTPInternalServerError(text=f"{type(exc).__name__}: {exc}")
        timings["total_s"] = round(time.perf_counter() - t0, 4)
        print(
            f"  seq {req['seq']:>5}  {arr.shape[0]:>3} actions  "
            f"prep {timings['prepare_s'] * 1e3:5.0f} ms  "
            f"infer {timings['infer_s'] * 1e3:5.0f} ms"
        )
        return web.Response(
            # Pinned to the width this checkpoint actually produces (measured
            # at load, not assumed), so a reshape that quietly disagreed with
            # the model would be caught here rather than on the rig.
            body=encode_chunk(
                arr,
                seq=req["seq"],
                timings=timings,
                action_dim=host.action_dim,
            ),
            content_type="application/octet-stream",
        )

    app.router.add_get("/meta", meta)
    app.router.add_post("/reset", reset)
    app.router.add_post("/act", act)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .../pretrained_model dir"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address (default loopback; reach it through an SSH tunnel). "
            "Anything else publishes an UNAUTHENTICATED inference endpoint."
        ),
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default=None, help="cpu/cuda (default auto)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    if not ckpt.exists():
        raise SystemExit(f"❌ checkpoint not found: {ckpt}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 loading {ckpt} on {device} ...")
    host = PolicyHost(str(ckpt), device)
    m = host.meta()
    print(
        f"  ✓ '{m['policy_type']}' policy: {m['n_obs_steps']} obs step(s) in, "
        f"{m['n_action_steps']} actions out, {m['action_dim']}-D"
    )
    print(f"  cameras: {', '.join(m['cameras'])}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"⚠️  binding {args.host}: this endpoint is unauthenticated.")
    print(f"🌐 http://{args.host}:{args.port}  (GET /meta, POST /reset, POST /act)")
    web.run_app(
        build_app(host), host=args.host, port=args.port, print=lambda *_a, **_k: None
    )


if __name__ == "__main__":
    main()
