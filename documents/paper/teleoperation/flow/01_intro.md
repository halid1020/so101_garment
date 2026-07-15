# Flow — Introduction

```
P1  why teleoperation quality matters for garment manipulation
    (demonstration data for learning)                    -> our setting
P2  the accessible-hardware context: SO-101 / LeRobot is a low-cost
    educational platform for students and hobbyists, but its default
    leader--follower data collection needs a second matched arm and ties
    the operator's posture to the follower; VR headset teleoperation is
    the convenient alternative, yet LeRobot has no native VR path (cite
    LeVR, which adds one). Motivates our Meta-Quest interface.  -> props
P3  the three properties of the SO-101 that shape the design:
    (a) 5 DoF / no wrist yaw -> 6-D tracking over-constrained;
    (b) small reach vs natural human gestures -> out-of-envelope is
        the norm;
    (c) position-controlled bus servos -> smoothness matters, expanded:
        discrete goal positions + finite stiffness mean jerky commands
        excite oscillation, saturate the bus, and wear gears
                                                        -> challenges
P4  NEW explicit challenges paragraph: names the five challenges this
    paper addresses, each with a forward link to the section that
    handles it — wrist under-actuation (5-DoF vs 6-DoF hand pose;
    method), cheap-servo command smoothness (filtering/prelim),
    small-arm reach envelope (envelope), dual-arm coordination
    (user study task), operator workload (clutch + user study)
                                                        -> contributions
P5  contributions list (6 items), each with a forward link; item 1
    briefly defines One-Euro filtering in one clause and forward-links
    to preliminaries; item 3 forward-links to the telegrip method
    section and appendix; item 4 is the NEW decoupled thumbstick wrist
    interface; item 6 defines NASA-TLX and SUS in one clause each
P6  reading guide: forward links to preliminaries, system, method,
    experiments, conclusion                              -> living-doc note
P7  living-document note (kept)
```
