# 3 Data-collection system — paragraph flow

P1  Intro paragraph: the system records a training-ready dataset plus a
    full-rate diagnostic log; this section walks the two-rate design ->
    stamp-on-read -> reference-time alignment -> drift budget -> stream set ->
    phase flag and gating. Forward links.
P2  Two-rate design: one dataset at a single integer frame rate holds the
    training-ready features; a separate full-rate side log holds every control
    signal with wall-clock stamps for offline analysis. Why two rates: a
    dataset has one rate, but diagnosis needs the fast signals.
P3  Stamp on read: each stream is timestamped at the instant it is read from
    the device (before decoding or resizing), on a monotonic clock, so
    alignment reflects capture time, not publication time.
P4  Reference-time alignment: at each frame the collector fixes one reference
    time and samples every stream there — nearest frame for images, linear
    interpolation for proprioception (spherical for orientation) — instead of
    grabbing each stream's latest value independently.
P5  Drift budget: for every stream and frame the collector records the residual
    between the reference time and the nearest real sample, and a live monitor
    shows it during collection; unsynchronised cameras cannot be re-phased in
    software, so the honest deliverable is a measured drift, not a claim of
    perfect sync. State the target budget relative to the frame period.
P5b Capture budget (why the bus, not just the clock, sets the drift): several
    colour streams share the host's USB controllers, and an uncompressed format
    does not fit — the cameras then negotiate a fraction of their requested rate,
    so the newest frame is already half an inter-arrival gap old and the residual
    of P5 is dominated by the capture rate rather than by clock skew. State the
    remedy (compressed transport, a queue short enough to bound staleness but
    deep enough not to starve the driver) and give the measured before/after.
P5c Exposure as a rate control: a second, independent way the capture rate is
    lost, and the one that survives fixing the transport. A camera cannot deliver
    frames faster than it exposes them, so an automatic exposure that lengthens
    in dim light drops the stream's rate without reporting anything. Argue that
    this bites the wrist views specifically — they look at a close, shadowed
    workspace — and that the dependence is on the LIGHTING, so the same rig
    yields different rates at different times of day. Remedy: fix the exposure,
    and note that the cost is small because the fixed value matches the
    brightness automatic exposure reaches in good light. Give the measured
    rate-versus-exposure relation. Hands over to the stream set.
P6  Stream set: enumerate what is recorded — colour views, aligned depth
    (stored losslessly outside the video pipeline because it is not
    three-channel colour), proprioception, measured end-effector pose, and both
    joint and end-effector targets. Units and layout match across streams.
P6b Episode identity: each episode carries a wall-clock identifier, because its
    position renumbers when any earlier episode is removed and so cannot name it;
    the identifier survives renumbering, keeping session logs and review
    decisions attached to the right recording.
P7  Phase flag and gating: a per-frame flag marks teleoperation-driven frames
    so non-teleoperation frames (homing) can be masked. Episode-level gating is
    GRADED, not absolute: a stream loss holds the episode and resumes it when the
    stream returns, abandoning it only if the stream stays away; a session fault
    still discards outright. Justify by the cost asymmetry (a blink vs a whole
    demonstration).
P6c Durability at episode granularity: each episode is committed to storage as
    it is accepted, rather than at the end of the session. Argue from the failure
    mode, not from tidiness: a dataset library that batches episode records in
    memory leaves a session unreadable while it runs and loses every unwritten
    record if the session is interrupted, and a recording whose record never
    landed is unusable for training even though its frames and video are on the
    drive. Committing per episode also makes a session reviewable while it is
    still being collected.
P7b Honesty about the recovery: a held episode contains a real gap that the
    index-derived timestamps do not show, so every hold is counted and reported
    with the episode for review, and any end the operator did not ask for sounds
    a cue (an operator watches the arms, not the terminal). Flag the waiting
    period as unjustified.
