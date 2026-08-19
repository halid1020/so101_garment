# Flow — System overview

```
P0  intro paragraph: hardware, software pipeline, pluggable IK,
    rehearsal harness — forward links                     -> paragraphs
P1  Hardware: rig geometry, the five joints, and the geometric fact
    used later, now EXPLAINED: (i) wrist_roll axis == gripper long
    axis, so roll about the tip is one joint; (ii) lift/elbow/flex
    axes are mutually parallel (all horizontal, normal to the arm
    plane), so the tip elevation angle is the SUM of those three joint
    angles up to fixed offsets — an affine function; armplane uses
    this to build reachable orientations, telegrip to invert the
    wrist analytically                                    -> servos
P2  servo model note (position control, twin gains)       -> software
P3  Software pipeline: reader thread, filtering, IK thread, per-arm
    joint threads; clutch semantics; button roles (enable to ready /
    park to rest / episode bracket / re-home without interrupting a
    recording); optional recording layer — fixed-rate episode dataset
    in a standard imitation-learning format plus a full-rate log of
    every control signal; the trigger drives the gripper
    through a CAPPED jaw opening (closed at full press, a bounded
    fraction of the range when released — fine garment pinching does
    not need the full splay); no code paths                -> IK layer
P3b Leader-arm interface: passive replicas, joint-to-joint tracking with no IK
    and no clutch, keyboard buttons, same capped jaw command so either input
    yields one demonstration format. Then WHY the per-joint velocity limit must
    be measured against elapsed time, not the control period: applied per
    iteration it becomes a per-iteration allowance, so a loop delayed by the
    host's own camera capture and video compression holds the followers to a
    fraction of the intended speed and the operator sees them trail — worst
    while recording. Capping the elapsed time one step may claim keeps a long
    delay from becoming a jump.
P3c A second route to the same lag, from the bus rather than the clock: a serial
    bus to the arms drops the occasional reply, which is ordinary, so what
    matters is the controller's answer to it. Retrying inside the tick costs a
    fraction of the period; abandoning the tick costs the whole of it and leaves
    the followers on the previous target, which is the very quantity P3b bounds.
    Note the asymmetry that hid this: the follower reads already retried and the
    leader read did not, so identical bus behaviour was visible on one side only
    and looked like a worse leader bus.                     -> IK layer
P4  Pluggable IK methods: narrow interface + adapter + rate limiter,
    described functionally (no class names); the native upstream
    Telegrip stack as an independent comparison path (no file paths)
                                                          -> rehearsal
P5  Rehearsal and benchmark harness: rehearsal and deployment share
    ONE control stack by construction (only the robot backend differs,
    so a method behaves identically in both); twin scene, cameras,
    mock device; the viewer now draws orientation triads for the
    MEASURED and COMMANDED gripper poses (so the operator sees attitude
    error directly); the idle-sag artifact fix (kept, reworded)
```
