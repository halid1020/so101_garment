# 5 Replicability — paragraph flow

P1  Intro paragraph: reproducing the platform needs its measured geometry, its
    camera calibration, its rates, and its action constants; this section lists
    what is measured and stored, and the readiness procedure. Forward links.
P2  Geometry and provenance: the rig geometry comes from one parametric CAD
    description shared with the digital twin, so the documented dimensions and
    the physical build cannot drift; the validation object is a printed cube of
    stated size.
P3  Camera calibration: the depth camera's intrinsics and depth scale are read
    from the device and stored with each dataset; the exterior cameras' fixed
    mounts and measured poses are recorded, because a shifted camera between
    collection and deployment breaks a policy silently.
P4  Rates and timing: the dataset rate, the full-rate side log rate, and the
    proprioception rate are stated, together with the measured drift budget
    from the collection section.
P5  Action constants: the stored action definition (units, joint order,
    gripper cap, scales, frame convention) is what an autonomous controller
    must reproduce; it travels with the dataset.
P6  Readiness procedure: a preflight check confirms the rig is at the fixed
    collection state — calibrated arms on stable aliases, protected cables,
    home poses, opened cameras and depth device, disk headroom, and the
    host-tuning items that reduce timing jitter.
P6b Shared-bus precondition: the check also reports which peripheral bus hub
    each sensor and the storage volume depend on, because a rig assembled from
    peripheral devices concentrates unrelated components onto shared hubs
    without anyone deciding to. Argue from the asymmetry: sensors sharing a hub
    costs a recoverable pause, whereas storage sharing a hub with the sensors
    recording into it removes the file being written, so the two deserve
    different verdicts. Give the operator the grouping, not just a warning,
    since the remedy is choosing a different socket.
P7  On-robot replay check: a recorded episode is played back on the arms, which
    closes the loop between what the dataset says and what the hardware does.
P8  Episode review and curation: the dataset is inspected in a browser tool that
    lists datasets and recordings and plays a selected one, and a flawed episode
    is removed there, with the dataset rewritten so curation leaves a valid
    dataset rather than a gap.
P8b Motion review: playback answers whether a demonstration did the right thing,
    not whether it did it smoothly, and jerkiness is a defect a policy imitates.
    That property is a DERIVATIVE, which the dataset does not store, so the
    review tool differentiates the recorded positions. Two claims, both measured:
    (i) the derivative is legible at the recorded rate rather than assumed to be
    — the encoder's resolution puts the acceleration noise floor about an order
    of magnitude below the motion being judged, so plain central differences
    suffice and a filter would blur the very transients being looked for;
    (ii) end-effector motion is reconstructed through the platform's own
    kinematic model, so Cartesian smoothness can be judged on recordings that
    stored no Cartesian stream — which is every recording driven from the leader
    arms, since the Cartesian channels exist only in the pose-tracking mode.
    Hand back to the argument that reviewing before training keeps a collection
    honest.
