# DreamZero — a world action model at rig scale

Implementation notes for `src/so101_policies/dreamzero/`: what it reproduces from
*World Action Models are Zero-shot Policies* (arXiv 2602.15922), what it does
not, and how to read what comes out.

## The idea, in one paragraph

A policy trained only to map observations to actions has to infer the dynamics
of the world implicitly, from action labels alone. A **world action model** is
trained to predict future *frames* as well, so it learns dynamics from every
consecutive frame pair rather than only from what an action label happens to
reveal. DreamZero denoises video and actions **jointly, under one flow-matching
objective**, which is what keeps them aligned — and the paper's own failure
analysis is that when it fails, the video prediction was usually wrong and the
actions faithfully executed it.

## What is reproduced

| paper | here |
|---|---|
| Joint flow matching over `[video latents ; actions]` (Eq. 2–3) | `modeling_dreamzero.forward` via `common/flow.py` |
| Chunk-wise teacher forcing, `C_k = {(z₁ʲ, a₁ʲ)}` for **j < k** (Alg. 1) | `masking.training_mask` |
| Between-chunk causal attention (Fig. 14) | `masking.py`, with training/inference masks proven equal |
| Autoregressive rollout, real observation replacing the predicted latent (Alg. 2) | `predict_future` |
| DreamZero-Flash decoupled schedules (Eq. 5–6) | `flow.sample_flash_times`, `--dreamzero-flash` |
| Savitzky–Golay chunk smoothing (App. D.3) | `common/smoothing.py` |
| All views tiled into ONE frame | `tile_cameras` |
| K = 2 latent frames, M = 4 chunks (App. C) | config defaults |

## What is not, and why it matters

These are limitations, not details. They belong in the paper's limitations
section, not a footnote.

- **No video pretraining.** The paper's central argument is that a WAM inherits
  physical priors from web-scale video; DreamZero is built on Wan2.1-I2V-14B.
  We train from scratch. **So this tests the objective and the architecture, not
  the prior** — which is the single biggest reason our absolute numbers are not
  theirs, and why the experiments are framed as testing the paper's *claims*.
- **A per-frame image VAE**, not Wan's temporal video VAE. Wan folds four raw
  frames into one latent frame; ours does not, so a latent frame is a frame. To
  keep a chunk's video and actions describing the same 1.6 s, the frames are
  **subsampled** (`config.frame_stride`) instead. Same interval, coarser
  visual sampling.
- **~1/100th the parameters** — about 59M at the defaults against 14B. The
  paper's own 5B ablation scored 21 % against 14B's 50 %, so scale is *known* to
  matter here and our numbers sit below both.
- **Language is a learned task embedding**, not a frozen text encoder. Every
  dataset on this rig carries one task string; a text encoder would be an
  expensive way to look up a constant.
- **adaLN modulation is per token, not per sample.** It has to be — Flash gives
  video and action tokens different timesteps inside one forward pass, and a
  single conditioning vector per sample cannot express that.

## The three things that are easy to get wrong

**1. The time direction.** There are two flow-matching conventions and they run
opposite ways. pi0.5 puts noise at t = 1 and integrates down; DreamZero's Eq. 2
puts the clean sample at t = 1 and integrates up. `common/flow.py` implements
pi0.5's, and every constant quoted from the paper is mirrored into it with the
paper's own number in the comment. A model written in one and sampled in the
other **trains perfectly and emits noise**.

**2. The teacher-forcing leak.** A predicted chunk must not attend to its own
clean twin — that block *is* the answer. Let them see each other and the model
learns to copy, scores a beautiful loss, and predicts nothing at inference where
the twin does not exist. The first draft of `masking.py` had exactly this bug;
it is why the mask is a separate module you can print and a test asserts the
invariant directly.

**3. Video/action time alignment.** A chunk's frames and its actions must cover
the same interval. The paper gets this free from a temporal VAE; we get it from
`frame_stride`, and `config` refuses a `chunk_size` that will not subsample
evenly. Misaligned, the objective learns to match a video of one moment with
actions of another — which trains, and is simply wrong.

## Running it

```bash
lerobot-train --policy.discover_packages_path=so101_policies \
              --policy.type=so101_dreamzero \
              --dataset.repo_id=... --dataset.root=...
```

or through the matrix, where `dreamzero` is a policy name like any other. The
driver adds the discovery flag itself; `--dreamzero-flash` selects the decoupled
schedule.

**The frozen VAE is fetched from the Hub on first use** (`stabilityai/sd-vae-ft-mse`,
83.7M parameters). A compute node with no outbound network needs it staged first,
the same way `lerobot/pi05_base` is — see `hpc/README.md`. It is excluded from
the checkpoint, since it is identical in all of them.

## Reading the output

`predict_action_chunk` returns actions, as any policy does. The world model's
distinctive output is `predict_future` / `predict_future_frames`: **what it
thinks will happen**, alongside what it plans to do about it.

That is what makes prediction accuracy measurable here in a way the paper never
reports — in the twin the ground-truth future exists, so predicted frames can be
compared against what actually happened. Report it **per camera and per horizon
step, against a held-last-frame baseline**: a tactile gel image is nearly static
until contact, so a predictor that simply copies the current frame scores well on
average, and without that bar on the chart the numbers flatter the model.
