# Flow — Preliminaries (new section)

Purpose: the common anatomy of Quest-to-arm teleoperation, so a reader
new to the area can follow the method section. Covers the shared data
flow, the retargeting mathematics, the common IK families, common
orientation/wrist mappings, and common out-of-envelope handling, for
single or dual arms with 5, 6 or 7 DoF.

```
P0  intro paragraph: what the section covers, forward links -> subsecs
S1  Data flow (subsection):
    P1  the canonical pipeline: tracked controller poses (~50-90 Hz)
        -> filter -> retargeting map -> IK -> joint commands
        (~100 Hz+); grips/triggers as discrete channel; define the
        frames (tracking frame, operator frame, robot base frame, EE
        frame) with notation used later
S2  Retargeting maps (subsection):
    P2  absolute vs relative (clutched) mapping; the clutch equation in
        its general form p* = p_E0 + s R (p_H - p_H0); why clutching
        exists (workspace mismatch, operator comfort); scaling s
    P3  orientation retargeting: direct 6-D tracking for >=6-DoF wrists;
        the under-actuation problem at 5 DoF (yaw coupled to base
        azimuth); the two standard escapes: relax/zero the orientation
        cost, or construct reachable targets
S3  IK families (subsection):
    P4  differential/QP IK (velocity-level, tasks + limits);
        damped least squares; per-tick nonlinear optimisation;
        analytic/split solutions for low-DoF chains; the trade-offs
        (accuracy vs smoothness vs solve cost vs branch jumps)
S4  Input filtering (subsection):
    P5  One-Euro filter definition (adaptive-cutoff first-order
        low-pass), its two knobs (min cutoff vs beta), and why it is
        the interactive-input default
S5  Workspace handling (subsection):
    P6  reach envelopes; what happens with no handling (solver
        saturates at limits); the standard remedies (projection,
        freezing, velocity shaping) that Section "envelope" makes
        concrete
```
