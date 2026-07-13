# Flow — Related work

```
P0  intro paragraph: four themes, forward-linked           -> P1..P4
P1  bimanual teleoperation of low-cost arms: joint-space
    leader-follower (ALOHA, Mobile ALOHA, GELLO) vs task-space
    VR/vision retargeting (AnyTeleop, Open-TeleVision); where our
    Quest-based task-space approach sits and why (no leader hardware,
    operator moves freely)                                -> P2
P2  controlling under-actuated (5-DoF) wrists: Telegrip's split IK on
    the SO-100 lineage; the vr-teleop-kit DLS family; how full 6-D
    trackers inherit the over-constraint; our armplane construction
    as the alternative                                     -> P3
P3  differential IK and task-space control: Pink/Pinocchio QP, mink,
    classical DLS, bounded NLS reference                   -> P4
P4  input filtering: One-Euro filter, its jitter/lag trade-off  -> P5
P5  workspace awareness: projection onto reachability
    approximations; our envelope + head-to-head policy comparison
```
