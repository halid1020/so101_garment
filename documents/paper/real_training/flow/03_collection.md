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
    Hands over to the stream set.
P6  Stream set: enumerate what is recorded — colour views, aligned depth
    (stored losslessly outside the video pipeline because it is not
    three-channel colour), proprioception, measured end-effector pose, and both
    joint and end-effector targets. Units and layout match across streams.
P7  Phase flag and gating: a per-frame flag marks teleoperation-driven frames
    so non-teleoperation frames (homing) can be masked; episodes that lose a
    stream or fault are discarded, so only clean episodes enter the dataset.
