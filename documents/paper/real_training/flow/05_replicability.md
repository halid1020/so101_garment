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
