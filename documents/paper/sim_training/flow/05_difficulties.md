# 5 Difficulties and resolutions — paragraph flow

Honest catalogue (guideline §1): one subsection per problem, each with
symptom → diagnosis → fix → what remains unjustified.

1. Intro paragraph: why this section exists (reproducibility; hidden
   fixes are defects) and how it is organised (contact, geometry,
   control, then collection, training, tooling).
2. Contact grasping journey: initial misses (jaws closing beside the
   cube) → diagnosed as azimuth-dependent steady-state IK error; naive
   pre-grasp servo wound up through pipeline lag and made things worse →
   converge-dwell + low gain + clip; the decisive fix was commanding the
   pinch *below* the cube centre so solver undershoot lands on it
   (visual review of failure animations led to this).
3. Place-phase dragging: longer place dwells *reduced* relay success —
   the hold servo kept correcting through the lag and dragged the
   already-placed cube; freeze-on-place resolved it.
4. Fixed simple-mode scenario: a simple-mode dataset is staked on one
   repeated scenario, so it inherits the oracle's *worst* case, not its
   average; the default relay scenario was place-limited (~1 in 4) and
   tripped the collection floor. Two-oracle differential test (lag-free
   analytic solves it every time, real-time teleop only sometimes) pins
   the cause on pipeline lag, not script/geometry; a servo-gain bump
   confirmed the negative result. Fix: select the fixed scenario by seed
   search (back-link to experiments).
5. Envelope floor versus table height: the twin's table top sits below
   the arm-base plane the workspace envelope assumes; table-level grasp
   targets violated the envelope's floor tuned for the real rig — a
   scoped per-scene override keeps the real stack untouched. Note the
   real-rig implication.
6. Mid-air hand-over abandonment: recorded as a design lesson (yaw
   drift + gripper clash), motivating the relay (back-link to tasks).
7. Renderer cache: one renderer cached per width only — two cameras at
   different resolutions silently reused the wrong buffer; keyed cache
   fix.
8. Non-monotonic training and checkpoint selection: success is not
   monotonic in training step — one checkpoint scored 100% while its
   neighbours grasped air (0%); reading only the last checkpoint would
   condemn the method. Fix: validate every checkpoint on held-out seeds
   and advance the best (ties → later step); an optimisation property,
   not a bug (back-link to experiments).
9. Training-stack gotchas: local dataset loading, normalisation living
   in processor pipelines rather than the policy, video backend
   bundling its own decoder, output directories that must not exist,
   the gated tokenizer licence, model sizing (full finetune of the
   large VLA does not fit prosumer memory; low-rank adaptation does),
   multi-camera activation memory (per-camera encoders at full
   resolution OOM — an activation not parameter cost, so LoRA cannot
   help; downsample before the encoders), and the pretrained VLA's fixed
   observation-key convention (rename map baked into the saved processor
   so eval inherits it).
10. Close: what remains unjustified (contact constants, servo gains,
   dwell durations) — flagged, not hidden.
