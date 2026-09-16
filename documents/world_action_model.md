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
reports — the ground-truth future is on disk, so predicted frames can be compared
against what actually happened:

```bash
venv/bin/python tool/eval_world_model.py \
    --checkpoint <run>/checkpoints/last/pretrained_model \
    --dataset <collection>/<dataset> --episodes 0-9
```

It reports **per camera and per horizon step, against a held-last-frame
baseline**, and both of those are load-bearing:

- **Per camera**, because the views do not behave alike. Four of five on this rig
  are tactile: a gel image is nearly static until contact, then changes fast. One
  averaged number hides exactly the moment worth predicting.
- **Against the baseline**, because "nothing changes" is a strong predictor of a
  static scene. Measured on the sim dataset, simply holding the last frame scores
  **67.8 dB** on the right wrist camera. A model reported without that bar looks
  excellent for the wrong reason, so the headline is not PSNR — it is *on how
  many horizon steps does the model beat holding*.

`untile_cameras` is what makes per-camera reporting possible: the views are tiled
into one frame for the model, and split back out for scoring, with prediction and
ground truth both compared at cell resolution so neither is upscaled to meet the
other.

## Status: trained, and what the probe cost to learn

MEASURED 2026-09-15 on thanos (RTX 3090 Ti, 24.5 GiB), sweeping batch upward
and stopping at the first size that fails — which never came. Batch 2, 4 and 8
all survived 400 steps:

| batch | `mem_gb` | 400 steps |
|---|---|---|
| 2 | 3.35 | 4:26 |
| 4 | 5.56 | 8:11 |
| 8 | 9.89 | 16:00 |

So the ceiling on this card is a lower bound and not a ceiling, and it is not
worth chasing: at 9.89 GB of 24.5 there is better than two-fold headroom.
**Memory is not what binds this policy.** Two other things are.

**Time.** 0.42 step/s in the probe, 0.77 step/s in the real run once startup is
excluded from the average — the probe's figure buys model construction and the
first dataloader fill inside only 400 steps, so it is the pessimistic one. At
0.77 step/s the matrix's 80 000-step default is about **25 hours**, against the
36 the row budgets. It fits, but not with much room.

**Power, which took the machine down.** A first attempt at eight loader workers
reached step 2000 in 46 minutes drawing **404 W of a 480 W limit**, and thanos
went down at 13:54 on 2026-09-16 — no traceback, no OOM, no Xid, no journal
entry, `who -b` the only record. That is the signature `train_destinations.yaml`
already records for the diffusion runs, and 404 W is *above* the 395 W peak of
the diffusion arm that trained through. This is the established power ceiling
reached by a new policy, not a new fault. There is no sudo on that box, so the
power limit cannot be lowered and the only lever is to feed the card less.

**Halving the loader workers did not lower the peak.** At four workers the
sampled peak was 415 W — if anything higher, and the step rate was unchanged at
0.77 step/s, so the workers were never the constraint. What actually protects
the run is not a mitigation but a recovery: a checkpoint every 2000 steps
(about 45 minutes) and a supervising loop that resumes from it, so a cut costs
under an hour instead of a day. A cut that takes the machine with it still
needs something outside the process to restart the loop.

### The one measurement that was not a measurement

The probe script queried `nvidia-smi` *after* each arm had exited, so it
recorded an idle 4 MiB card three times and reported that as the VRAM cost. The
numbers in the table above come from lerobot's own `mem_gb` metric field, which
is logged at full precision on every metric line and was in the log all along.
Worth stating plainly because the failure was silent and plausible: a number
appeared, it was small, and nothing about it looked wrong.

**Stage the VAE before submitting anywhere.** The frozen
`stabilityai/sd-vae-ft-mse` (~330 MB) is fetched from the Hub on first use and
deliberately kept out of the checkpoint, while every launcher exports
`HF_HUB_OFFLINE=1`. On a compute node with no internet that fails *after* the
GPU is reserved. `hpc/README.md` has the login-node command.

## Where the result shows up

`tool/eval_world_model.py` writes `prediction.json` beside the checkpoint it
scored — under `checkpoints/<step>/prediction/`, not under `checkpoints/last`,
because `last` is a symlink and the score belongs to the checkpoint that
produced it. The console's Training tab reads it into a **Future prediction**
panel, ordered by the verdict rather than by PSNR.

That ordering is the whole point. Holding the last observed frame scores 67.8 dB
on a near-static camera, and four of this rig's five cameras are gel images that
barely move until contact. A tactile camera at 40 dB that loses to holding on
every horizon step has taught the model nothing; the overhead camera at 30 dB
that wins has. A panel sorted by PSNR would put those two in the wrong order and
present the model's worst camera as its best result.

## The other world model, and why it could not be compared

`harena_fastwam` is the opposite trade. It is a real ~6 B Wan-class model with a
genuine pretrained video prior, where DreamZero is 59 M parameters trained from
scratch — so between them they bracket the question this rig actually wants
answered: does a video prior buy anything at rig scale, or does the data do it?

That comparison was not possible, for a reason that had nothing to do with either
model. `wan.modular.infer_joint` decodes both future video and actions and
returns `{"video": …, "action": …}` — and **nothing in the policy API ever called
it**. `predict_action_chunk` reaches `infer_action` and only that. So the port
owned a video prediction it could not be asked for, while DreamZero could be
asked and owned no prior. `tool/eval_world_model.py` rejected FastWAM outright.

`harena_fastwam_predict` closes that: a subclass in a sibling package, the same
shape as the crop variants, so the ported files stay byte-identical to upstream
and `tool/port_policies.py --check` stays green at 18 files. It adds the three
calls the scorer makes — `predict_future_frames`, `tile_cameras`,
`untile_cameras` — plus the two context fields it reads. FastWAM already tiles
its cameras by concatenating them along width, so the tiling pair is a rename of
what it does, not a new representation.

Two details in it are load-bearing, and both would be quiet rather than loud if
they were wrong:

- **The first predicted frame is dropped.** `infer_joint` pins
  `latents_video[:, :, 0:1]` to the conditioning frame's latents at every
  denoising step, so frame zero of the output is the input reproduced. Scoring it
  would measure the model on an image it was handed, and would flatter every
  horizon curve at step one — exactly where the held-last-frame baseline is
  already strongest.
- **The sampler is pinned** (`predict_seed`, `predict_inference_steps`).
  `infer_joint` samples, so two passes over one observation disagree; an arm
  compared against another arm under an unpinned sampler measures the sampler.
  Same reasoning as `analysis.diffusion.plan`, and the same mistake was made
  there first.

The context arithmetic differs between the two and the scorer has to know which:
DreamZero counts context in latent CHUNKS, FastWAM conditions on a single first
frame. `n_context_chunks` and `latent_frames_per_chunk` are both 1 on the
subclass — not a convention chosen for convenience, but the number of observed
frames the prediction is conditioned on, which is what the scorer slices off
before comparing.

**Still untrained.** The subclass makes FastWAM *scoreable*; it does not make it
*trained*. That needs ~20 GB of staged weights (`Wan-AI/Wan2.2-TI2V-5B`, its
Diffusers VAE, UMT5-XXL, `lerobot/fastwam_base`) and a card larger than thanos's
24.5 GiB — CREATE with `SBATCH_CONSTRAINT=h200`, where `freeze_video_expert`
leaves only the ~1 B action expert training. If it does not fit even there, that
is a measurement to record as a ceiling, not something to retry blind.
