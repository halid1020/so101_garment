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
P5  Sharing a peripheral bus: separate the THREE things this conflates, because
    conflating them sent us after the wrong cause for weeks.
    (a) Bandwidth contention is real and is why capture is compressed.
    (b) A stream running below its requested rate is NOT evidence of contention:
        ours measured the same rate alone as under full load, and the cause was
        exposure (P6), not the bus. State this as a correction — the earlier
        account of this platform attributed it to bandwidth.
    (c) The failure that actually costs a session is a whole hub going, which
        takes every device behind it at once. Argue the asymmetry: a stream lost
        is recoverable, because the recorder holds the episode and resumes it,
        but the same hub carries the volume being written to, so the fault does
        not interrupt the session, it removes what the session is being written
        into. Point forward to the three answers — a topology precondition
        before, one diagnosis grouped by hub after, per-episode durability so
        what was recorded survives. Flag: the simultaneous-stream ceiling is
        machine-specific and measured, not derived.
P6  Appearance drift, and the stronger reason underneath it: automatic exposure
    and white balance make the same scene look different across episodes and
    between collection and deployment. Then the finding that outranks it —
    exposure bounds frame rate, because no camera delivers frames faster than it
    exposes them, so automatic exposure lengthening in dim light silently costs
    capture rate. That is the mechanism P5(b) was mis-attributing. Resolution is
    the same either way, fixing exposure and recording the value, but the
    justification is now measured rather than aesthetic; cross-reference the
    capture budget rather than repeating its numbers.
P6b Episode boundaries in a shared video file: episodes are packed end to end and
    the recorded window's end is EXCLUSIVE, so the frame sitting on it is the
    next recording's first. Anyone re-implementing a reader meets this; state it
    as a property of the format, with the symptom it produces when missed.
P7  Host timing jitter: background load and power governors cause frame
    overruns; resolution is the readiness check's host-tuning items and the
    overrun warnings. Flag any thresholds set by convention.
P8  Coverage for autonomy: clean expert trajectories under-represent the
    recovery states an autonomous policy will visit; resolution is a
    collection protocol that deliberately varies object poses and includes
    recoveries. This is a data-design obligation, not a code fix.
