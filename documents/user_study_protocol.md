# Bimanual teleoperation user study — runbook

Operational protocol for the user study designed in
`documents/paper/teleoperation/` §"Bimanual user study (protocol)". The paper fixes
the *design* (conditions, measures, analysis); this file is the
step-by-step script for actually running sessions. Both objective
(logged/timed) and subjective (questionnaire/interview) evaluation are
collected, so the study pairs with the simulation benchmark
(`documents/teleop_benchmark_results.md`).

**Status: designed, not yet run.** Update this file and the paper with
results when sessions happen (living-paper rule in CLAUDE.md).

---

## 1. Overview

- **Participants:** ~5 (pilot scale). Mixed prior VR/robot experience is
  fine — record it in demographics. No motor impairments that prevent
  holding Quest controllers for ~1 h.
- **Design:** within-subjects; every participant uses all 4 conditions on
  the same task; condition order counterbalanced (Latin square, §4).
- **Session length:** ~75 min per participant.
- **Roles:** 1 experimenter (runs software, holds the torque-disable
  switch, times trials), optionally 1 note-taker.

## 2. Conditions

| id | pipeline | how to launch |
|---|---|---|
| C1 | armplane, `pink_full`, OOE `project` | `venv/bin/python tool/meta_quest_teleopration.py --method pink_full --oob-mode project` |
| C2 | armplane, `pink_relaxed`, OOE `project` | `venv/bin/python tool/meta_quest_teleopration.py --method pink_relaxed --oob-mode project` |
| C3 | armplane, `telegrip` split IK, OOE `project` | `venv/bin/python tool/meta_quest_teleopration.py --method telegrip --oob-mode project` |
| C4 | upstream Telegrip stack (own UI + IK, no envelope) | `venv/bin/python tool/telegrip_native.py --autoconnect` (see `documents/telegrip_native.md`) |

Do **not** reveal which condition is "ours"/"the baseline"; refer to them
as A/B/C/D to the participant (map A–D to C1–C4 per the Latin square).

## 3. Task

**Primary — pick–handover–place** (3 timed trials per condition):

1. A 40 mm soft foam cube sits on the RIGHT start marker (tape cross,
   ~20 cm in front of the right arm base).
2. Right arm picks the cube.
3. Handover to the left gripper over the rig center line (both grippers
   must hold the cube simultaneously at some instant — observer checks).
4. Left arm places the cube on the LEFT target marker (15 cm left of
   center). Cube fully inside the 8 cm target square = success.
5. Timeout 3 min. A drop may be retried within the timeout from where the
   cube lands (count the drop).

**Secondary — towel half-fold** (1 trial per condition, only if the
session is on schedule): 30×30 cm towel flat on the table; both grippers
grasp the two near corners and fold the towel in half onto the marked
line. Success = top edge within 25% of bottom edge everywhere (visual).

## 4. Condition order (balanced Latin square)

| participant | order |
|---|---|
| P1 | C1 C2 C4 C3 |
| P2 | C2 C3 C1 C4 |
| P3 | C3 C4 C2 C1 |
| P4 | C4 C1 C3 C2 |
| P5 | C1 C2 C4 C3 (repeat of P1) |

## 5. Session script

1. **Setup (before participant arrives):**
   - `source setup.sh`; confirm arms respond (`tool/check_mirror.py` / `tool/fit_joint_offsets.py`
     if in doubt: follower_0 = RIGHT, follower_1 = LEFT).
   - Charge Quest; confirm headset ↔ host network for C1–C3 reader and
     for C4's browser UI (`https://<host>:8443`).
   - Lay tape markers (start cross, center line, 8 cm target square,
     towel fold line). Cube + towel at hand.
   - Test-launch each condition once; for C4 accept the self-signed cert
     in the headset browser beforehand.
   - Prepare per-participant data sheet (§6) and questionnaire forms (§7).
2. **Consent + demographics (5 min):** purpose, right to stop anytime,
   what is recorded; age range, handedness, prior VR / teleop / gaming
   experience (Likert 1–5 each).
3. **Familiarization (10 min):** `venv/bin/python tool/quest_sim_teleop.py
   --method pink_relaxed` — explain clutch (hold both grips), re-grip
   ("point both handles down when gripping — every grip recalibrates"),
   triggers, height lock, that the headset can be placed anywhere. Free
   play until the participant can move both EEs deliberately.
4. **Per condition (~12 min × 4):**
   1. Launch the condition (map A–D per Latin square). For C1–C3 note the
      console is being logged (§6).
   2. ~3 min free practice with the cube (not scored).
   3. Three recorded trials: experimenter says "go", starts stopwatch,
      stops at place/timeout. Record the objective sheet per trial.
   4. Towel-fold trial if on schedule.
   5. Participant fills NASA-TLX, SUS, and the 6 custom items (§7) —
      about the condition just used, while it is fresh.
   6. Short break, water; arms to rest pose.
5. **Closing (10 min):** forced ranking of A–D (best→worst overall), then
   semi-structured interview (§8). Thank + debrief (reveal conditions if
   asked).

## 6. Objective data

Per trial, on the data sheet:

| field | source |
|---|---|
| completion time (s) | stopwatch (start "go" → cube released on target) |
| success (Y/N) + failure mode | observer (drop / timeout / unrecoverable pose) |
| drops (count) | observer |
| re-grips (count) | console: each grip prints a recalibration line (C1–C3); observer count in C4 |
| OOE time (s) | console: envelope margin in the 2 s debug print (C1–C3 only; write "n/a" for C4) |
| rig collisions (count) | observer |

Console capture: launch C1–C3 with `... 2>&1 | tee
outputs/user_study/P<n>_<cond>.log` so re-grip and OOE lines are
recoverable afterwards. Optionally record each trial as a LeRobot dataset
episode for post-hoc EE smoothness analysis with `src/sim_benchmark/metrics.py`.

## 7. Subjective instruments (per condition)

- **NASA-TLX (raw)** — six 0–100 scales: mental demand, physical demand,
  temporal demand, performance, effort, frustration.
- **SUS** — the standard 10 statements, 1–5 agree scale (score per the
  usual 0–100 formula).
- **Custom teleop-feel items** (1 = strongly disagree … 7 = strongly
  agree):
  1. The grippers went where I intended (precision).
  2. The wrist followed my hand rotation without noticeable delay
     (wrist agility).
  3. The system behaved predictably when I reached the edge of the arms'
     range (boundary behavior).
  4. Releasing and re-gripping to re-center my hands was easy (clutch).
  5. I could position myself/the headset where I wanted and still control
     comfortably (headset-anywhere).
  6. Coordinating both arms at the same time felt manageable (bimanual).

## 8. Interview guide (semi-structured, ~10 min, recorded or noted)

1. Walk me through how the whole process felt, from putting on the
   headset to finishing a trial.
2. What felt most natural? Least natural?
3. Was there a moment you lost confidence in what the robot would do?
   What happened?
4. Did you notice differences between A–D? Which and how did you adapt?
5. How did the two-arm coordination feel compared to what you expected?
6. If you could change one thing about the controls, what?

## 9. Analysis plan

- Objective: per-condition median + IQR of time/success/drops/re-grips/
  OOE time; Friedman test across conditions, Wilcoxon signed-rank
  post-hoc — *exploratory only* at n=5.
- Subjective: TLX and SUS descriptive stats per condition; custom items
  plotted per item; rankings tallied.
- Interviews: thematic coding (two passes, one coder + one checker).
- Write results into `documents/documents/paper/teleoperation/sections/05_experiments.tex`
  (user-study subsection) and cross-link from
  `documents/teleop_benchmark_results.md`.

Per-participant session packs (consent/demographics, per-condition trial
sheets and questionnaires in that participant's Latin-square order, and
the closing interview sheet) live in `documents/user_study/` — print one
pack per participant and file it after the session.

## 10. Safety & ethics checklist

- [ ] Torque-disable (Ctrl+C / kill switch) within experimenter's reach
      at all times; cleanup path is crash-safe (`so101_dual_arm.py`).
- [ ] Participant briefed they may stop at any time without reason.
- [ ] Workspace clear of bystanders; arms cannot reach the participant's
      standing position.
- [ ] Breaks offered between conditions (VR fatigue/nausea check —
      stop the session on any nausea report).
- [ ] Data stored pseudonymously (P1…P5); recordings only with consent.
