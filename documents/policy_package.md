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
