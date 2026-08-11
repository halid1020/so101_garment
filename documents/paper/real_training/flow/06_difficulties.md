# 6 Difficulties — paragraph flow

P1  Intro paragraph: an honest record of the obstacles met and how each was
    resolved, so the platform is reproducible and its remaining weak points are
    visible. One short subsection per difficulty.
P2  Temporal skew across streams: grabbing latest values skews frames;
    resolution is stamp-on-read plus reference-time alignment plus a logged
    drift budget. Residual: unsynchronised cameras keep an irreducible
    cross-camera skew that we measure but cannot remove in software.
P3  The stale-command labelling trap: when no fresh command exists the action
    falls back to the measured state, which can teach a spurious "hold" on
    moving homing frames; resolution is a per-frame teleoperation flag and a
    logged fallback rate so those frames are maskable.
P4  Depth cannot use the colour video pipeline: sixteen-bit depth through a
    three-channel encoder is corrupted; resolution is lossless per-frame depth
    stored beside the dataset with its scale and intrinsics.
P5  Shared-bus bandwidth: many compressed streams on shared buses drop frames;
    resolution is compressed capture, per-stream rate and drop counters, and
    discarding episodes with a lost stream. Flag: exact simultaneous-stream
    ceiling is machine-specific and measured, not derived.
P6  Appearance drift: automatic exposure and white balance make the same scene
    look different across episodes and between collection and deployment;
    resolution is locking exposure where the device allows and logging the
    settings.
P7  Host timing jitter: background load and power governors cause frame
    overruns; resolution is the readiness check's host-tuning items and the
    overrun warnings. Flag any thresholds set by convention.
P8  Coverage for autonomy: clean expert trajectories under-represent the
    recovery states an autonomous policy will visit; resolution is a
    collection protocol that deliberately varies object poses and includes
    recoveries. This is a data-design obligation, not a code fix.
