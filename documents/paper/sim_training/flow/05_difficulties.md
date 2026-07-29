# 5 Difficulties and resolutions — paragraph flow

Honest catalogue (guideline §1): one subsection per problem, each with
symptom → diagnosis → fix → what remains unjustified.

1. Intro paragraph: why this section exists (reproducibility; hidden
   fixes are defects) and how it is organised (contact, geometry,
   control, tooling).
2. Contact grasping journey: initial misses (jaws closing beside the
   cube) → diagnosed as azimuth-dependent steady-state IK error; naive
   pre-grasp servo wound up through pipeline lag and made things worse →
   converge-dwell + low gain + clip; the decisive fix was commanding the
   pinch *below* the cube centre so solver undershoot lands on it
   (visual review of failure animations led to this).
3. Place-phase dragging: longer place dwells *reduced* relay success —
   the hold servo kept correcting through the lag and dragged the
   already-placed cube; freeze-on-place resolved it.
4. Envelope floor versus table height: the twin's table top sits below
   the arm-base plane the workspace envelope assumes; table-level grasp
   targets violated the envelope's floor tuned for the real rig — a
   scoped per-scene override keeps the real stack untouched. Note the
   real-rig implication.
5. Mid-air hand-over abandonment: recorded as a design lesson (yaw
   drift + gripper clash), motivating the relay (back-link to tasks).
6. Renderer cache: one renderer cached per width only — two cameras at
   different resolutions silently reused the wrong buffer; keyed cache
   fix.
7. Training-stack gotchas: local dataset loading, normalisation living
   in processor pipelines rather than the policy, video backend
   bundling its own decoder, output directories that must not exist,
   the gated tokenizer licence, and model sizing (full finetune of the
   large VLA does not fit prosumer memory; low-rank adaptation does).
8. Close: what remains unjustified (contact constants, servo gains,
   dwell durations) — flagged, not hidden.
