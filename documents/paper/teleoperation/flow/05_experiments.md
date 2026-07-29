# Flow — Experiments

Organised around research questions: each subsection states one RQ and
reports the experiment that answers it. Numbers stay reproducible from
the table set; the thumbstick interface is evaluated only at rehearsal
level (scripted mock device), so it is discussed qualitatively and
pointed to the results log, not promoted into the benchmark tables.

```
P0  intro paragraph: the digital-twin setup (physics/control rates,
    clutch-faithful mock trajectories) and the four research questions,
    each forward-linked to its subsection                  -> metrics
P1  Metrics paragraph (kept): position/orientation/roll-lag/jerk/etc.
S1  RQ1 — does position-first IK track table-plane strokes better than
    full 6-D tracking on a 5-DoF arm? Tracking suite (kept content),
    Table circle_r5cm; headline pink_full vs pink_relaxed vs scipy_ls
    vs telegrip.
S2  RQ2 — which method best follows a commanded wrist orientation
    without wrecking position? Wrist-agility suite (kept content),
    Table wrist_roll_f1hz; ADD one paragraph: the decoupled thumbstick
    interface trades absolute hand-attitude tracking for direct
    joint-space wrist control; a rehearsal sweep (results log, not the
    benchmark suite) shows the hold variant holds attitude steady and
    keeps the best position tracking, at the cost of no longer
    following the hand — an intent the open-loop orientation metric
    cannot score.
S3  RQ3 — how should infeasible (out-of-envelope) targets be handled?
    Out-of-envelope suite (kept content), Table envelope_slide.
S4  RQ4 — does the system hold up end to end, and do the objective
    winners survive a human operator? End-to-end rehearsal (kept) +
    the user-study protocol (kept), framed as the subjective RQ.
KEEP verbatim: the ACT/Diffusion training-experiment TODO (needs real
    policy runs, cannot be resolved by writing).
```
