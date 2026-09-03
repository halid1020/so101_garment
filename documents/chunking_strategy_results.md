# Splicing action chunks over a network link — measured results

The numbers behind the chunking subsection of the simulation-training paper.
The paper argues from these; this file records how they were produced, so a
disagreement can be settled by re-running rather than by argument.

Companion runbook: `documents/remote_policy_inference.md` (how to stand the
pieces up). Sweep harness: `tool/run_policy_sim.py`. Tables regenerated with
`tool/export_chunking_table.py`.

## What was measured, and against what

| | |
|---|---|
| Policy | diffusion, `diffusion_handover_20260805_112423`, checkpoint `030000` |
| Selected by | validation seeds (100% on its 10-seed pool), 73% on the EVAL pool |
| Task | `handover` (the relay whose plate both arms can reach) |
| Trained at | 30 fps, camera 640x480, cameras `scene`/`wrist_left`/`wrist_right` |
| Chunk | 32 actions, 2 observation steps per request |
| Scenario | seed 14, repeated — see below |
| Rate | `--fps 30`, matching the dataset |
| Inference | a separate machine (RTX 3090 Ti), reached over ssh port-forwarding |
| Environment | the rig laptop, MuJoCo twin, `--pace realtime` |

**Why one repeated scenario and not a held-out pool.** This checkpoint was
trained in the overfit-one-scenario mode: 100 demonstrations of seed 14 and
nothing else. Asking it about a scenario it never saw would measure a
generalisation failure that no splice can cause or cure. Holding the scene
fixed leaves the splice as the only thing that varies between cells, which is
the comparison this experiment exists to make. It also means these results say
nothing about how a splice behaves on a policy that generalises.

## The grid

Seventeen cells, from `--grid all`. A cell is a strategy TOGETHER with the
parameters it reads, because the strategies are not comparable as bare names —
`common/chunk_sweep.py` carries the argument.

| strategy | family | cells |
|---|---|---|
| `sync` — hold for the reply | blocking | 1 |
| `append` — queue behind | aligning (nominally) | 1 |
| `replace` — discard and replace | aligning | 1 |
| `receding` — execute a fraction, then replan | blocking | 4 (0.25, 0.5, 0.75, 1) |
| `blend` — cross-fade over a window | aligning | 6 (w3/w5/w10 x linear/exp) |
| `ensemble` — weighted average | aligning | 4 (0.3, 0.5, 0.7, 0.9) |

**`rtc` (real-time chunking) is not in the grid, and this is a scope boundary
rather than an omission.** Its smoothing happens inside the denoiser on the
inference host, so it is a distinct method only when the host performs it.
`tool/policy_server.py` does not advertise `rtc`, and against a host that does
not, the client-side splice is exactly `replace`. The client refuses the run
rather than silently downgrading it, so a row labelled `rtc` cannot be a
mislabelled `replace`. Implementing it host-side is separate work.

## Reproducing

```bash
# on the GPU machine
venv/bin/python -u tool/policy_server.py --port 8770 \
  --checkpoint outputs/vla_sim_long/diffusion_handover_20260805_112423/\
simple/handover/diffusion/checkpoints/030000/pretrained_model

# on the rig laptop: a tunnel that comes back after a blip, then the sweep
ssh -N -L 8770:127.0.0.1:8770 <gpu-host>
MUJOCO_GL=egl PYTHONPATH=.:src venv/bin/python -u tool/run_policy_sim.py \
  --task handover --server http://127.0.0.1:8770 --grid all \
  --seeds simple --simple-seed 14 --episodes 10 --fps 30 \
  --camera-map wrist_camera_left=wrist_left,wrist_camera_right=wrist_right \
  --out outputs/chunking/stage_a.md --resume
```

Give the sweep its **own port**. The host keeps one session, handed out by the
last reset, and answers every other request with `409`; a second client
claiming that slot costs an episode even though the sweep now recovers from it.

## Measurement conditions, and one that went wrong

The round trip is a property of the link AND of what else is using the
inference GPU. It must therefore be reported per cell, not once for the run.

**Three cells ran under a contended GPU, and it matters less than it looks.**
While `receding@0.25`, `@0.5` and `@0.75` were running, checkpoint evaluations
for a different experiment shared the same card, and the median round trip rose
from roughly 750 ms to roughly 1300 ms. `receding@0.75` straddles the change:
its first four episodes saw 1264-1465 ms and its last two saw 646-648 ms.

For a BLOCKING splice this does not touch the result. The client blocks inside
the tick, so when it returns there is always an action to command and the twin
advances exactly one step: **simulated time does not pass during a round trip**,
which is why these cells report no held ticks at all. Nothing the twin does
depends on the wall clock, so success, seam ratio, path length and placement
error are all latency-independent. What the contention corrupts is the
wall-clock column, and only there.

For an ALIGNING splice the round trip IS load-bearing — the offset at which the
new plan is grafted on is derived from the measured trip in ticks, so a slower
link changes what the strategy does. Every aligning cell (`append`, `replace`,
and all of `blend` and `ensemble`) ran under a quiet link at 652-798 ms, so the
cells that need consistent conditions have them.

The wall-clock figures for the three contended cells are therefore marked as
measured under contention rather than quietly compared with the rest.

## Results

*Pending: the table lands here when the sweep and its corrective pass finish.*

## What is not established here

* One policy, one architecture, one task, one scenario, one link.
* Ten episodes per cell in the first pass, which separates families but does
  not resolve two tunings of the same strategy against each other.
* A blocking strategy's cost is wall-clock time. A simulator makes that cheap;
  a rig with a person waiting does not, and nothing here prices that.
