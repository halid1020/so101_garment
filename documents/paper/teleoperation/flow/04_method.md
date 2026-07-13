# Flow — Retargeting method

```
P0  intro paragraph: the four pieces (operator frame, clutched
    position mapping, armplane orientation, envelope), forward links;
    names the pipeline `armplane` (formerly "production")  -> subsecs
S1  Operator control frame (kept, minor wording)
S2  Clutch (expanded):
    P1  DEFINE clutch: a deadman-style engage/disengage control — the
        robot follows only while both grips are held; releasing
        freezes the arms and re-gripping re-anchors
    P2  the clutched mapping equation (kept)
    P3  pros/cons vs non-clutched (absolute) mapping:
        + workspace ratcheting (reach beyond arm workspace in steps)
        + safe pause / operator repositioning without robot motion
        + every re-grip is a fresh calibration (recovers from a bad
          first calibration)
        - no persistent absolute correspondence (position drift
          between hand and robot frames accumulates by design)
        - repeated grip cycles add operator workload (measured by the
          user study's clutch item)
        ⚠ unjustified: grip threshold value flagged as convention
S3  Armplane orientation construction (kept; add explicit statement
    of what happens at engagement — see S5)
S3b Roll ratcheting subsection: the clutch already ratchets roll
    (re-anchoring at every grip); the ~320° wrist-roll joint is the
    ceiling; jaw-equivalence rewrap at engagement edges restores
    headroom (blend glides it, trigger suppresses it); mid-hold hint;
    per-hand joystick-click reset to neutral; guard band flagged
    unjustified
S4  Workspace envelope + policies (kept)
    P+  operator feedback paragraph: debounced OOE cues (edge, repeat,
        stop; intensity ∝ penetration depth); terminal bell today,
        controller vibration specified but awaiting the headset-app
        rebuild; repeat period flagged unjustified
S5  NEW subsection: gripper orientation at teleoperation start —
    what happens to the gripper orientation the moment the clutch
    engages, stated explicitly for every method:
    P1  at the rising edge: operator frame derived; per-hand handle
        axis captured (handles held plumb-down => "handle-down =
        gripper-down"); roll reference (knuckle axis) captured from
        the gripper's CURRENT roll
    P2  the EE rotation at engagement is latched and the commanded
        orientation is blended from it to the constructed target over
        a fixed time by geodesic interpolation (slerp), so engagement
        never jerks the wrist; position is anchored at the EE pose at
        grip, so the gripper does not translate either
    P3  method-by-method: armplane + all registry methods receive the
        same blended target through the same input processing; the
        telegrip method then sets its wrist joints analytically from
        that target (same start behaviour, different tracking after)
S6  Telegrip split IK (kept)
```
