# `src/so101_policies/` — every policy, in this repo

Policies used to live in LeRobot and be selected by name. They now live here,
and are still selected by name — the difference is that we own the code and can
change it. This describes how that works, what a *port* is and is not, and how
to add a policy.

## Why it works without a second trainer

LeRobot has a plugin seam, and it is load-bearing:

1. `lerobot.scripts.lerobot_train.train` is `@parser.wrap()`-decorated.
2. `parser.wrap` reads any `--<field>.discover_packages_path=<pkg>` argument and
   calls `load_plugin` on it **before** draccus parses anything
   (`configs/parser.py:293-301`).
3. `load_plugin` imports the package. Our `__init__.py` imports every policy,
   and `@PreTrainedConfig.register_subclass("...")` fires as a side effect.
4. `get_policy_class` falls through to `_get_policy_cls_from_policy_name`
   (`policies/factory.py:165`), which resolves the class from that registry.
5. `make_pre_post_processors` has the same fallback (`factory.py:640`).

So this trains a policy defined here:

```bash
lerobot-train --policy.discover_packages_path=so101_policies \
              --policy.type=so101_act ...
```

and everything downstream — `tool/policy_server.py`, `tool/run_policy.py`,
`hpc/fetch_policies.sh`, `src/common/analysis/`, the console's Training tab —
works unchanged, because they all resolve a policy the same way.

`test/system/long_vla_real.sh` adds the discovery flag automatically for any
policy whose name begins `so101_`; nothing needs to be passed by hand.

## The naming contract

LeRobot derives the policy class and the processor factory from the **config
class name**, by string surgery (`factory.py:606-637`). It is not a convention
you may bend:

| file | must contain |
|---|---|
| `so101_policies/<x>/configuration_<x>.py` | `So101<X>Config`, decorated `@PreTrainedConfig.register_subclass("so101_<x>")` |
| `so101_policies/<x>/modeling_<x>.py` | `So101<X>Policy`, with `config_class` and `name = "so101_<x>"` |
| `so101_policies/<x>/processor_<x>.py` | `make_so101_<x>_pre_post_processors(config, dataset_stats=None)` |

A policy must also subclass `PreTrainedPolicy` and implement its five abstract
methods (`get_optim_params`, `reset`, `forward`, `predict_action_chunk`,
`select_action`), and its `__init__` must tolerate `dataset_stats=` and
`dataset_meta=` kwargs, which `make_policy` passes when it has them.

## What a port is

`act`, `diffusion` and `pi05` are **ports, not rewrites**. The upstream module
tree moved into this repo unchanged. That is deliberate and it buys two things:

- **Checkpoints stay interchangeable.** An identical module tree means identical
  `state_dict` keys, so the finished 80 000-step ACT checkpoint loads into either
  implementation and plans the *same actions, bit for bit*. That is measured on
  the real file in `test/integration/test_policy_ports_checkpoints.py`, not
  assumed. Nothing already trained was invalidated by the move.
- **And they train the same.** The action comparison above runs under
  `torch.no_grad()` in `eval()` mode, so it exercises none of the code that
  decides how a run *trains* — dropout, ACT's VAE sampling, diffusion's noise
  and timestep draws, and the loss itself are all on the training branch. The
  same test therefore also compares `policy.forward(batch)`: the loss, and
  every gradient it produces, bit for bit on shared weights. MEASURED on the
  finished ACT checkpoint: loss 0.492358922958374 from both, and 153 identical
  parameter gradients.

  That still says nothing about what `lerobot-train` assembles *around* the
  model — the processor pipeline the factory derives from the config class
  name, the optimiser and scheduler presets, the dataloader, and the plugin
  discovery that has to resolve one of our type strings before draccus parses
  anything. So `tool/compare_port_training.py` runs the real trainer twice from
  the same seed and compares the logged loss step for step. MEASURED on thanos
  against the real five-camera dataset, 60 steps at batch 2 on the card:

  | policy | vs | agreement |
  |---|---|---|
  | `act` | `so101_act` | identical at all 6 logged points |
  | `diffusion` | `so101_diffusion` | identical at all 6 logged points |

  The log prints the loss at three decimals, so that is an agreement to a
  thousandth over many steps; the integration test is the other way round — one
  batch, compared bit for bit. Neither replaces the other.
  `make test-port-parity DATASET_ROOT=<ds>` is the CPU version, in the system
  tier.

  `so101_pi05` is covered by the source comparison only: it is a 4.1B model
  that needs its 14.5 GB base to instantiate at all, which is more than a test
  tier should reach for.
- **A LeRobot bump is one command.** Every rule the port applies lives in
  `so101_policies/_port.py`; `tool/port_policies.py` re-derives all nine files
  and `--check` reports drift. `test/unit/test_policy_ports.py` fails the moment
  one is hand-edited, including in pi0.5, which is far too large to instantiate
  in a test.

The port changes exactly four things, and nothing else may change:

1. relative imports (`from ..pretrained import`) become absolute, because only
   the module's home moved;
2. the registered type string gains its `so101_` prefix;
3. the config and policy class names gain their `So101` prefix, since LeRobot
   derives everything else from them;
4. **one non-cosmetic rename**: `ProcessorStepRegistry` is a *global* namespace,
   so pi0.5's `pi05_prepare_state_tokenizer_processor_step` collides with
   upstream's the moment both are imported — which is exactly what the
   equivalence test does. Ours is `so101_pi05_`-prefixed.

Old names are kept as aliases at the foot of each file, so the module bodies
read as they do upstream.

The three ported directories are **excluded from black/isort/flake8/mypy** in
`.pre-commit-config.yaml`. Reformatting them to this repo's profile would
destroy the diff against upstream that makes the equivalence claim checkable,
for no gain. Code we actually write is linted like everything else.

If a port ever needs a real edit, it stops being a port: take the file out of
`_port.PORTS` and say why.

## FastWAM — a port from a commit the pin has never seen

`so101_fastwam` is the fourth port, and the only one whose source is not in the
pinned checkout: FastWAM landed upstream after `LEROBOT_COMMIT` (3dd19d04), so
`../lerobot` does not contain it. Rather than move the pin — which would change
act, diffusion and pi05 underneath every finished checkpoint and invalidate the
byte-comparison the three existing ports rest on, including the port-parity
measurement on thanos — `_port.PORT_REF` names the commit to read that one
directory out of. A SHA and not a branch: `origin/main` moves, and a port that
silently re-derives from a different upstream on every fetch is not a port.

`tool/port_policies.py --check` re-derives it byte for byte exactly as it does
the others, so it is no less checkable for being read through `git show`.

Two things are particular to it:

* **It carries a subpackage.** FastWAM's model lives in `wan/`, seven files
  whose imports are already absolute or package-relative, so they are copied
  unrewritten (`_port.PORT_EXTRA`) — but they are still in `--check`, because
  being unrewritten is not the same as being unowned. Its `__init__.py` is
  *not* carried: upstream's re-exports `FastWAMPolicy`, which the port renames
  without adding an alias, so carrying it verbatim breaks the import.
* **The pinned `lerobot.processor` is missing exactly two names** —
  `make_default_policy_processor_steps` and `make_policy_processor_pipelines`,
  both added to `processor/factory.py` after the pin. Everything else FastWAM
  touches exists at the pin, name by name, MEASURED. Those two are reproduced
  line for line in `common/processor_compat.py`, and the port rewrites that one
  import to point at them — the same mechanical move `_port._RELATIVE` already
  makes for `..pretrained` and `..utils`. Editing the file instead would stop it
  being a port at all. A test asserts the pin *still* lacks them, so the day it
  gains them the shim can be deleted rather than quietly outliving its reason.

The bare `fastwam` name stays in `matrix.POLICIES` and stays refused by the
module probe, which is the honest report: the port is what can be trained today.

**Before launching one**, note that FastWAM pulls a 5B Wan video backbone
(`Wan-AI/Wan2.2-TI2V-5B`) and a `umt5-xxl` text encoder. It is a far larger
model than anything else in the run matrix, the 24.5 GiB box may not hold it at
any batch, and both weights must be warmed on a login node because every
launcher exports `HF_HUB_OFFLINE=1`.

## The cropped-tactile variants

`so101_act_crop`, `so101_diffusion_crop` and `so101_pi05_crop` are each a
two-field subclass of the corresponding port. They exist because Grad-CAM on the
finished five-camera ACT checkpoint shows the policy attending to the **edges**
of the fingertip images — `right_arm_right_gripper` saturates along its left
edge and top-right corner while the gel centre stays cold — and it does so in
frames where nothing is in contact. That is the signature of light leaking in at
the gel boundary, a known failure of vision-based tactile sensors: the border is
bright, it tracks the room rather than the object, and a network will learn it.

**The crop is a processor step, not a model change.** `common/tactile.py` holds
`So101TactileCropProcessorStep`, registered `so101_tactile_crop`, inserted at
**index 0** of the twin's preprocessor. Index 0 is load-bearing:
`RenameObservationsProcessorStep` is step 0 of every one of these pipelines, and
on pi0.5 it renames the rig's cameras onto openpi's slot names — a crop placed
after it would look for camera names that no longer exist and silently do
nothing at all. The preprocessor runs during training as well as inference, so
this changes what the policy is trained on.

**It crops the centre and resizes straight back**, so no shape anywhere changes.
That is the decision the whole design rests on:

* ACT's per-camera token count stays 300, so `analysis/streams.py` keeps tiling.
  It assumes every camera contributes the same token width, and cropping only
  the fingertips would break that assumption for the very analysis these runs
  are measured by.
* The diffusion encoder sizes its feature dimension from a dummy input at
  construction; an unchanged input shape cannot disagree with it.
* pi0.5 letterboxes with `resize_with_pad`, so a changed aspect ratio would
  change how much padding each slot gets — a second, uncontrolled difference
  between a cropped run and its baseline.

The cost is a little interpolation blur, which is the right trade when the goal
is removing a border rather than gaining resolution.

**A separate registered type rather than a flag on the twin**, because the crop
has to travel with the checkpoint: a run trained cropped must be *served*
cropped, and LeRobot rebuilds the processor pipeline from the policy type. A
flag would let the two drift apart silently. `tactile_crop = 1.0` is the
uncropped control and is bit-identical to the twin — the crop returns the input
untouched rather than making a no-op interpolation pass, so the control arm is
the same run and not a third condition.

**The fraction is measured, and measuring it changed the design.**
`tool/measure_tactile_border.py` takes per-pixel temporal standard deviation and
mean luminance over sampled frames. Run against
`fold-short-from-flattend-tactile` (six episodes, one frame in twenty) it found
two things that the obvious "crop the border" reading does not survive:

| camera | safe row crop | safe col crop | edge brightness rise |
|---|---|---|---|
| `left_arm_left_gripper` | 0.62 | **1.00** | +12.2 % |
| `left_arm_right_gripper` | 0.45 | 0.82 | +10.9 % |
| `right_arm_left_gripper` | 0.53 | 0.81 | +7.4 % |
| `right_arm_right_gripper` | 0.55 | **1.00** | +22.2 % |

*Safe* is the tightest centred crop keeping every line whose temporal variation
is in the frame's top quartile — the lines where the gel actually responds.

1. **The rim is a smooth vignette, not a band.** Luminance falls monotonically
   from both edges to the middle, so there is no boundary to find and no
   threshold at which the answer stops moving. The tool prints a sensitivity
   row for that reason; a fraction quoted without it is an opinion wearing a
   measurement's clothes.
2. **Horizontally, the bright rim and the responsive region are the same
   pixels.** On two of the four sensors the most active columns run to the frame
   edge. A centred width crop cannot remove the leak without removing signal.

So the default crops **rows only** — `(0.80, 1.00)`, comfortably inside the 0.62
bound on the tightest camera and removing the brightest rows on all four. The
first version of this cropped both axes at 0.7; that would have cut away part of
what the sensors were reporting, and only the measurement said so.

**No contact gate.** Feeding tactile only once contact is established was
considered and left out: "stable contact" is a temporal predicate, training
shuffles frames, and this repo's own contact segmentation
(`analysis/phases.py`) needs a whole episode because it thresholds on quantiles
of that episode's own gripper channels. A training-time gate would have to be
re-derived from the tactile image against a no-contact reference, with its own
threshold to justify — separate work, not a flag on this.

## `flowmatch` — the control for the world model

`so101_flowmatch` is not a port. It is pi0.5's objective and action expert on the
diffusion policy's vision trunk, and it exists to be a **control**: it shares
DreamZero's training objective (conditional flow matching over an action chunk)
and shares nothing else — no video prediction, no world model, no pretrained
video prior. The gap between the two therefore measures the world-modelling
objective with everything else held fixed, which is a cleaner comparison than
either paper runs against its own baselines.

The vision trunk is `DiffusionRgbEncoder` imported from
`so101_policies.diffusion` — that literal class, not a reimplementation — so
"same backbone" is a fact rather than a claim. Its eight config fields are
spelled exactly as the diffusion policy spells them, because the encoder reads
them off our config object.

### The time convention, which is the thing to get right

There are two flow-matching conventions in the literature and they run time in
**opposite directions**:

| | t = 0 | t = 1 | sampling |
|---|---|---|---|
| pi0.5 / openpi (`modeling_pi05.py:744`) | clean action | noise | integrates *down* |
| DreamZero (arXiv 2602.15922, Eq. 2) | noise | clean sample | integrates *up* |

A model trained in one and sampled in the other **trains perfectly and emits
noise** — the loss curve looks entirely healthy the whole way.
`so101_policies/common/flow.py` implements pi0.5's and says so at the top;
anything else that uses it must state its own direction at the call site.

MEASURED on a policy overfitted to one batch for 1500 steps: **9.1 %** of target
scale at 5 inference steps in the right direction, **359.9 %** in the wrong one.
`test/unit/test_flow_matching.py` guards the algebra cheaply (integrating the
true velocity is exact in one step, because the path is straight);
`test/integration/test_flowmatch_learns.py` guards the trained network, with the
wrong-direction control included so the assertion cannot pass vacuously.

## Reading a checkpoint through the other implementation

A checkpoint records the type that trained it, and that name picks the class.
`tool/retarget_checkpoint.py` writes a directory whose weights are **symlinked**
and whose `config.json` names the other member of the pair:

```bash
venv/bin/python tool/retarget_checkpoint.py \
    --checkpoint outputs/policies/<run>-act --to so101_act --out <dir>
```

This is how a repo-local pi0.5 gets something to finetune: `lerobot/pi05_base`
says `pi05`, so `--policy.path` to it would load LeRobot's class however the run
was asked for. The driver retargets it automatically for `so101_pi05` — 14.5 GB
of weights symlinked, one field rewritten. It refuses to retarget between two
policies that are not a ported pair, because the weights would not fit.

## Adding a policy

1. Write the three files under `src/so101_policies/<x>/` to the contract above.
2. Import it in `src/so101_policies/__init__.py` — the import *is* the
   registration.
3. Add an entry to `POLICIES` in `src/common/training/matrix.py` with `steps`,
   `batch`, `hours`, `"local": True` and `"module": "so101_policies.<x>"`. The
   CLI's `--policies` help, the console's checkboxes, the refusals and the
   resolved defaults all follow from that one entry.
4. Add its sizing arm to the `case` in `test/system/long_vla_real.sh`, and any
   flags it needs beside the `diffusion`/`pi05` blocks. The discovery flag and
   the dispatch loop already handle it.
5. Widen the `case` in `hpc/submit_real.sh`.

`policy_available` probes the module rather than comparing versions, so a policy
whose package is missing is refused before a GPU is reserved — and, because the
entry is marked `local`, it is told to check the checkout rather than to bump
LeRobot.
