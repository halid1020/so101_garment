# What each input stream contributes

The rig records five cameras — one overhead RGB and four fingertip tactile
cameras — plus twelve joint values. A policy trained on all of them plans well
enough; what none of that says is **which of those streams it is actually
using**. This is how that is measured.

There are two different questions here and they are easy to run together.

| question | how it is answered | what it costs |
|---|---|---|
| What *could* a policy learn from each stream? | Train one per camera set and compare (`hpc/runs.tsv`) | GPU-days |
| What does *this* trained policy use? | Attribution on one checkpoint (`tool/analyse_policy_inputs.py`) | Minutes |

They can disagree, and where they do that is the finding rather than a fault: a
stream whose absence a retrained policy compensates for is not the same as a
stream the trained policy ignores. Report both.

## The one structural fact everything rests on

Both policies trained here reduce their inputs to a vector in which **each
stream owns a contiguous piece**.

**ACT** builds its transformer encoder input as
`[latent, state, cam₀ × H·W, cam₁ × H·W, …]`, cameras in `config.image_features`
order — the checkpoint's own order, not alphabetical. At 480×640 a ResNet-18
`layer4` map is 15×20, so each camera owns **300 consecutive tokens** and five
cameras plus latent and state make **1 502**. Verified against the loaded
backbone, and `check_layout` asserts the spans tile the vector with no gap and
no overlap — an off-by-one here does not raise, it silently attributes the tail
of one camera to the head of the next and the figure still looks plausible.

**Diffusion** concatenates `[state, cam₀ feat, cam₁ feat, …]` per observation
step and flattens, so a camera owns a contiguous run *within each step block*
and the blocks repeat `n_obs_steps` times.

`src/common/analysis/streams.py` is that map, and it is pure: a config in, index
ranges out, no torch and no GPU.

## The four methods

**Occlusion** (`--method occlusion`) replaces one stream and re-infers. It is
behaviour rather than an inference about behaviour, so it is the ground truth
the others are scored against, and it needs no assumption about architecture.
One forward pass per stream — fast enough to run across a whole episode.

*Leave one out* replaces a single stream: a large effect means dependence, but a
small one does **not** mean the stream is unused, only that the others carry the
same information. *Only one in* (`--direction only_one_in`) replaces every other
stream and measures sufficiency instead. With four fingertip cameras there is a
lot of redundancy, so both directions are worth having.

**Integrated gradients** (`--method ig`) is the primary quantitative method,
because of its completeness axiom: the attributions sum to `f(x) − f(baseline)`.
That is what makes five cameras' shares five parts of one whole rather than five
unrelated numbers.

**Grad-CAM** (`--method gradcam`) says *where in the frame*, at the resolution
the network reasons at, for one backward pass.

**Attention** (`--method attention`, ACT only) reads the decoder's
cross-attention, which gives something nothing else here can: which stream was
consulted when planning action 0 of the chunk versus action 99. Diffusion has no
equivalent — it conditions a UNet on a concatenated vector with no attention
over cameras at all.

## Three measured things that change how the output is read

**Attention is nearly uniform, so the raw mass means almost nothing.** Each
camera owns 300 of 1 502 tokens, so uniform hands it 0.1997 — and the observed
shares are 0.192 to 0.208, within 1.5 % of that at most frames. Over the same
frames occlusion finds `central` worth ten times any fingertip, and integrated
gradients agrees with occlusion at a rank correlation of **+1.00**, while raw
attention ranks the five almost backwards at **−0.71**. The **deviation** from
uniform agrees at **+0.70**, so that is what is reported and plotted. What does
carry information is how attention varies along the chunk's own horizon:
`central`'s share runs about 0.31 for a plan's first actions and 0.08 for its
last, which no token count explains.

**Integrated gradients needs a fine path here.** The completeness error falls
0.29 (16 steps) → 0.22 (32) → 0.15 (64) → 0.017 (128) → 0.008 (256). The axiom
holds; this model is simply non-linear enough that a coarse Riemann sum misses.
The per-stream *shares* converge far sooner than the sum — the largest moves by
0.0016 between 64 and 128 steps — so 64 is the default and the completeness
error is returned with every result rather than assumed away. Raise `--ig-steps`
for a figure that has to carry the axiom.

**The baseline is part of the result.** There is no neutral image. Zeroing is
conventional and off-manifold — a policy has never seen a black frame, so its
reaction says as much about surprise as about dependence. Four are offered and
every figure names the one it used:

| baseline | what it isolates |
|---|---|
| `zeros` | the conventional choice; off-manifold |
| `mean` (default) | same brightness, no structure: texture against illumination |
| `blur` | coarse layout kept, fine detail gone: is a gel camera read for contact texture, or only for "something is there"? |
| `shuffle` | a real frame from elsewhere: fully on-manifold, so it asks "does it matter *which* frame this is" |

Where two baselines disagree about a stream, that disagreement is the honest
answer.

## Running it

Over recorded episodes — ground truth actions, many frames, no rig needed:

```bash
venv/bin/python tool/analyse_policy_inputs.py \
    --checkpoint outputs/policies/fold-short-from-flattend-tactile__all-act \
    --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile \
    --episodes 0-4 --method occlusion,ig,attention \
    --out outputs/analysis/act-all
```

Over a real rollout:

```bash
venv/bin/python tool/analyse_policy_inputs.py \
    --checkpoint outputs/policies/<run> --run outputs/policy_runs/<stamp>
```

A rollout is only analysable if it was recorded with **`--log-frames`**
(`tool/run_policy.py`). That flag is off by default because the frames are the
largest part of an observation; without them there are no pixels to attribute a
plan to, and the tool says so rather than producing an empty study.

Cost, measured on a laptop RTX 3050 with the five-camera ACT checkpoint: about
1.4 s a frame for occlusion, 16 s for integrated gradients at 64 steps, and
0.43 GB of VRAM. Occlusion is the workhorse for a whole episode; the gradient
methods are for chosen moments.

## What comes out

`attribution.json` holds every number, so a figure can be redrawn without
loading a checkpoint. Beside it:

1. **`*_over_time.png`** — each stream's share across the episode, with the
   gripper phases shaded behind it. The headline: does tactile rise at contact
   and fall away during the approach?
2. **`*_streams.png`** — streams by effect at one frame.
3. **`*_joints.png`** — stream × joint, rows normalised, gripper columns marked.
4. **`*_within_chunk.png`** — attention against position in the plan.
5. **`*_gradcam.png`** — Grad-CAM over each camera's frame.
6. **`*_methods.png`** — every method's shares side by side. Disagreement is the
   point of the figure, not a defect in it.

## Not fooling yourself

Saliency methods are easy to misread, so the toolkit carries its own checks.

- **Completeness** is a unit test, not a hope: if the attributions do not sum to
  `f(x) − f(baseline)`, the integration path is wrong.
- **`--sanity`** runs the model-randomisation test (Adebayo et al., *Sanity
  Checks for Saliency Maps*, 2018): re-run with randomised weights, and a method
  whose answer barely moves is measuring the input rather than what the model
  learned. The ACT checkpoint passes it in the strongest way available —
  randomised weights plan the same chunk whatever they are shown, so every share
  falls to exactly zero. It is destructive to the copy in memory; reload after.
- **Occlusion is the referee.** IG and attention are reported with their rank
  correlation against the measured occlusion effect, so a method that disagrees
  with behaviour says so on the figure instead of being quietly believed.
- **Phases come from the data.** An attribution averaged over an episode answers
  almost nothing — tactile cameras cannot contribute while the grippers are
  still travelling. The episode is segmented from the gripper channels
  themselves (columns 5 and 11), and every threshold is a quantile of that
  episode's own values: these grippers work in a narrow band whose position
  depends on the object and the day's calibration, so an absolute threshold
  would put every frame in one phase.

## An early result

On the finished five-camera ACT checkpoint, one episode of
`fold-short-from-flattend-tactile`: early in the episode the overhead camera and
proprioception carry almost everything and the four fingertips together
contribute about **3 %**; by mid-episode the tactile share reaches **20 %**,
then falls below 1 % as the arms withdraw. That is the shape the question was
asked to find, and it is what the trained ablation rows should now be compared
against.
