# Flow — Abstract

```
S1  what the paper is about: the teleoperation system for a bimanual
    low-cost 5-DoF rig driven by Meta Quest controllers   -> the setting
S2  the core difficulty: 5-DoF arms cannot track full 6-D poses, reach
    is small, servos want smooth commands                 -> our answer
S3  the answer, qualitatively: clutched retargeting with a
    reachable-by-construction orientation mapping (armplane), adaptive
    input filtering, a pluggable IK layer, an analytic workspace
    envelope with selectable out-of-envelope policies     -> evaluation
S4  how it is evaluated, qualitatively: a digital-twin benchmark
    (tracking, wrist agility, out-of-envelope) and a user-study
    protocol; full implementation detail in the appendix  -> end
```

Rules applied: no numbers, no performance figures, general-roboticist
level, active voice, focus on the teleoperation system (not the rig).
