# 2 Collection platform — paragraph flow

P1  Intro paragraph: the platform is a dual five-degree-of-freedom rig with a
    leader-follower option and a multi-view sensor suite; this section walks
    arms -> operator interfaces -> sensing -> host and buses. Forward links.
P2  Arms: two five-degree-of-freedom follower arms, a fixed left/right base
    layout shared with the digital twin; each arm carries a wrist camera.
P3  Operator interfaces: a head-mounted controller (Cartesian teleoperation
    through inverse kinematics) and a pair of kinematically identical leader
    arms (direct joint teleoperation). Same recorded semantics from either.
P4  Sensing: per side two visual-tactile fingertip cameras and one wrist
    camera; a central depth camera overlooking the workspace; the colour
    streams and the aligned depth stream are the policy's exterior views.
P5  Host and buses: everything reaches one laptop over two powered hubs, with
    the head-mounted controller on a direct link; a note that shared-bus
    bandwidth is a real constraint on simultaneous compressed streams, handled
    later by the drift monitor rather than assumed away. Now also state the
    ceiling itself, because with the fingertip cameras attached the suite
    exceeds it: each controller admits only a few compressed streams, and the
    readiness check counts a selection against its buses before a session opens
    anything. Cross-reference the difficulties section for the failure MODE,
    which is refusal rather than degradation.
P6  Final-state readiness: before collection the rig must be at a fixed, known
    state (calibrated arms on stable port aliases, protected cables, home
    poses, opened cameras); a preflight check confirms this and the platform is
    measured for reproducibility (forward link to replicability).
