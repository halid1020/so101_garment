# Flow — Retargeting method

```
P0  intro paragraph: the pieces (operator frame, clutched position
    mapping, armplane orientation, engagement, roll ratcheting,
    envelope, telegrip split IK, thumbstick wrist interface), forward
    links; names the pipeline `armplane`                   -> subsecs
S1  Operator control frame (kept, minor wording)
S2  Clutch (kept):
    P1  DEFINE clutch: deadman-style engage/disengage
    P2  the clutched mapping equation (kept)
    P3  pros/cons vs non-clutched (absolute) mapping; grip threshold
        flagged unjustified
S3  Armplane orientation construction (kept) + ADD: while the target
    is out of envelope the tip-azimuth reference is FROZEN, so the
    commanded roll (orthogonalised against the tip) does not drift as
    the saturated arm slides along the boundary; forward-link envelope
S4  Gripper orientation at engagement (kept)
S5  Roll ratcheting and the wrist-roll limit (kept)
S6  Workspace envelope + policies (kept), with:
    P.geom  ADD: the table-clearance floor now sits SLIGHTLY BELOW the
            table plane (a small negative value) so the gripper can
            actually touch the surface, rather than a few mm above it;
            reason = table-contact grasps need it, safety cost is one
            servo tick of penetration bounded by the same clamp
    P.feedback  operator feedback paragraph, EXTENDED: the audible cue
            is now a real speaker tone on the control computer as well
            as the terminal bell, with a per-arm pitch (lower tone =
            left arm, higher = right) so the operator hears WHICH arm
            left the envelope; same debounced edge + throttled-repeat
            behaviour; controller vibration still the intended endpoint;
            repeat period flagged unjustified
S7  Telegrip-style split IK (kept)
S8  NEW subsection: decoupled thumbstick wrist interface (`mymethod`):
    P1  motivation from operator report: fine wrist_roll/wrist_flex
        control via absolute hand orientation proved hard on the real
        rig; joint-space rate trims from the thumbsticks bypass both
        IK task conflict and wrist-axis coupling
    P2  what it does: the relaxed Pink POSITION solver plus clutched
        joint-space wrist trims — stick x -> wrist_roll rate, stick y
        -> wrist_flex rate, integrated in joint space; deadzone + expo
        shaping; DEFINE the trim clutch: while a stick is past its
        deadzone that arm's other joints freeze exactly and the handle
        is ignored; a forward-kinematics floor guard rejects flex
        increments that would push the tip below the table; on release
        the wrist stays put and the hand-to-robot correspondence
        re-anchors so handle control resumes with no jump
    P3  three selectable attitude behaviours between trims: HOLD
        (default; attitude held from grip, changed only by the sticks),
        the incremental wrist mapping with trims on top, and the weak
        absolute follow where trims decay back toward the hand attitude
    P4  pros/cons vs the armplane mapping: + clutched precision +
        start-anywhere; - loses direct hand-attitude correspondence
    P5  unjustified flags: deadzone 0.15, rates 60/45 deg/s, expo 2.0
        (initial engineering guesses, not hardware-tuned) and the trim
        direction signs (unverified on the real robot)
```
