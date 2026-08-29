"""One checkpoint, held open, answering "what would you plan from this?".

The seam every method in this package shares. It is deliberately NOT
``common.policy_client``: that one owns a queue, a session and a chunking
strategy because a rollout needs them, whereas an attribution needs one
observation to give one chunk, repeatably, with nothing remembered in between.

``predict_action_chunk`` is what makes that possible -- it returns the whole
plan and touches no queue -- and for ACT it is deterministic: the VAE latent is
sampled only ``if self.training``, so at eval it is zeros and two identical
observations give identical chunks. Diffusion is not; see
:mod:`common.analysis.diffusion`.

Actions come back UNNORMALISED, in the units the arms are commanded in
(degrees, and gripper open fraction), because a per-joint number is meant to be
read. The post-processor unnormalises one step at a time -- handing it a whole
chunk is the obvious thing to try and is wrong -- so a chunk is pushed through
it step by step, exactly as ``tool/policy_server.py`` does.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from lerobot.utils.constants import OBS_IMAGES

from common.analysis.streams import (
    act_layout,
    camera_names,
    check_layout,
    diffusion_layout,
    streams_of,
)


class Inference:
    """A loaded checkpoint and the shapes it implies."""

    def __init__(self, checkpoint: str, device: "str | None" = None, task: str = ""):
        import torch

        from tool.eval_sim_policy import build_batch, load_policy

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # from_pretrained treats a relative path as a Hub repo id and fails with
        # a message about repo names, which says nothing about the real cause.
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        if not os.path.isdir(self.checkpoint):
            raise SystemExit(f"❌ no checkpoint directory at {self.checkpoint}")
        self.policy, self.pre, self.post, self.type = load_policy(
            self.checkpoint, self.device
        )
        self.policy.eval()
        self._build_batch = build_batch
        self.task = task
        self.cfg = self.policy.config
        self.cameras = camera_names(self.cfg)
        self.streams = streams_of(self.cfg)
        self.action_dim = int(self.cfg.output_features["action"].shape[0])
        self.n_action_steps = int(getattr(self.cfg, "n_action_steps", 1) or 1)
        self.n_obs_steps = int(getattr(self.cfg, "n_obs_steps", 1) or 1)
        self.image_hw = tuple(list(self.cfg.image_features.values())[0].shape[1:])

    # -- what the policy's conditioning looks like -------------------------
    def tokens_per_camera(self) -> int:
        """MEASURED, not predicted: run one frame through the real backbone.

        ``replace_final_stride_with_dilation`` quadruples this and nothing in
        the declared image shape says so, and a token count that is wrong by any
        amount does not raise -- it silently attributes the tail of one camera to
        the head of the next, and the figure still looks plausible.
        """
        backbone = getattr(getattr(self.policy, "model", None), "backbone", None)
        if backbone is None:
            return 0
        height, width = self.image_hw
        with self.torch.no_grad():
            probe = self.torch.zeros(1, 3, height, width, device=self.device)
            feature_map = backbone(probe)["feature_map"]
        return int(feature_map.shape[-2] * feature_map.shape[-1])

    def camera_feature_dim(self) -> int:
        """Diffusion's per-camera encoder width, from the model rather than a guess."""
        encoder = getattr(getattr(self.policy, "diffusion", None), "rgb_encoder", None)
        if encoder is None:
            return 0
        if isinstance(encoder, (list, tuple)) or hasattr(encoder, "__getitem__"):
            try:
                return int(encoder[0].feature_dim)
            except (TypeError, IndexError, AttributeError):
                pass
        return int(getattr(encoder, "feature_dim", 0))

    def layout(self):
        """Where each stream sits in this policy's conditioning, and whether it tiles."""
        if self.type == "act":
            per_camera = self.tokens_per_camera()
            spans = act_layout(self.cfg, per_camera)
            total = 1 + sum(s.width for s in spans)
            return spans, total, check_layout(spans, total, latent=1)
        spans = diffusion_layout(self.cfg, self.camera_feature_dim())
        total = sum(s.width for s in spans)
        return spans, total, check_layout(spans, total)

    # -- the forward pass ---------------------------------------------------
    def batch(self, state: np.ndarray, images: "dict[str, np.ndarray]") -> dict:
        """One observation -> the batch the policy expects, normalised.

        A policy with more than one observation step was trained on ADJACENT
        frames, so a single observation is repeated to fill the window. That is
        an approximation and it is stated rather than hidden: for a still scene
        it is exact, and for a moving one it removes the velocity cue, which is
        itself worth knowing when a diffusion attribution reads oddly.
        """
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise ValueError(
                f"this checkpoint needs {', '.join(missing)}, which the "
                f"observation does not have (it has {', '.join(sorted(images))})"
            )
        batch = self._build_batch(state, images, self.task, self.device)
        batch = self.pre(batch)
        # The saved pre-processor carries the device it was TRAINED with -- cuda,
        # for every checkpoint here -- and moves the batch there regardless of
        # what this process asked for. On a machine with a GPU that is a silent
        # half-move: the weights are where we put them and the batch is not, and
        # the forward pass dies deep inside a Linear with a device mismatch that
        # names neither. Put it back where the weights are.
        batch = self.to_device(batch)
        if self.n_obs_steps > 1:
            batch = self._stack_window(batch)
        return batch

    def to_device(self, batch: dict) -> dict:
        """Every tensor in the batch onto this run's device. Lists included."""
        out = {}
        for key, value in batch.items():
            if self.torch.is_tensor(value):
                out[key] = value.to(self.device)
            elif isinstance(value, list) and value and self.torch.is_tensor(value[0]):
                out[key] = [v.to(self.device) for v in value]
            else:
                out[key] = value
        return out

    def _stack_window(self, batch: dict) -> dict:
        """Repeat one observation into the window a multi-step policy expects."""
        out = dict(batch)
        for key, value in batch.items():
            if not self.torch.is_tensor(value):
                continue
            if key.startswith("observation."):
                out[key] = value.unsqueeze(1).repeat_interleave(self.n_obs_steps, dim=1)
        return out

    def chunk(self, state: np.ndarray, images: "dict[str, np.ndarray]") -> np.ndarray:
        """The plan this observation produces: ``(n_action_steps, action_dim)``."""
        return self.chunk_from(self.batch(state, images))

    def chunk_from(self, batch: dict) -> np.ndarray:
        """As :meth:`chunk`, from an already-built batch (so it can be perturbed)."""
        torch = self.torch
        with torch.no_grad():
            planned = self.policy.predict_action_chunk(batch)
            if planned.ndim != 3:
                planned = planned.unsqueeze(0)
            planned = planned[:, : self.n_action_steps, :]
            steps = [self.post(planned[:, i, :]) for i in range(planned.shape[1])]
            out = torch.stack(steps, dim=1).squeeze(0)
        return np.asarray(out.detach().to("cpu"), dtype=np.float32).reshape(
            -1, self.action_dim
        )

    # -- the same forward pass, differentiable ------------------------------
    def chunk_tensor(self, batch: dict):
        """The plan as a tensor with gradients attached. Normalised units.

        ``predict_action_chunk`` is decorated ``@torch.no_grad()`` on both
        policies, so a gradient method cannot use it and has to call the model
        underneath by the same route the wrapper does -- including the stacking
        of ``OBS_IMAGES``, which is where the camera ORDER is fixed and so is not
        a detail that may be reproduced loosely.

        The output is left NORMALISED. The post-processor is affine, so
        unnormalising would only rescale every attribution by a constant per
        channel, and doing it per step would put a hundred extra nodes in the
        graph for no change in what is ranked.
        """
        torch = self.torch
        batch = dict(batch)
        keys = list(self.cfg.image_features)
        if self.type == "act":
            batch[OBS_IMAGES] = [batch[k] for k in keys]
            return self.policy.model(batch)[0]
        # Diffusion stacks along a camera axis instead of using a list, and its
        # sampler is stochastic -- see common.analysis.diffusion for the seed.
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in keys], dim=-4)
        return self.policy.diffusion.generate_actions(batch)

    def image_keys(self) -> "list[str]":
        """The batch keys the cameras arrive under, in the policy's own order."""
        return list(self.cfg.image_features)

    def describe(self) -> str:
        return (
            f"{self.type} on {self.device}: {len(self.cameras)} camera(s), "
            f"{self.n_obs_steps} obs step(s) in, {self.n_action_steps} actions "
            f"out, {self.action_dim}-D"
        )
