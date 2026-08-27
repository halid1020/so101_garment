# 4 Teleoperation-to-autonomy contract — paragraph flow

P1  Intro paragraph: this section states the principle that makes a
    teleoperated dataset usable for autonomy, then its two instances (joint and
    end-effector policies) and the metadata that ties them down. Forward links.
P2  The principle: the recorded action must be the command the robot actually
    followed — after rate limiting, joint clamping, the capped gripper mapping,
    and the workspace-envelope projection — not the raw operator input; and at
    inference the policy output must pass back through that same command path.
    Otherwise executed motion leaves the training distribution.
P3  Why the target, not the measured pose: the recorded intent is a future goal
    the arm moves towards, kept strictly separate from the measured state; a
    fallback rule defines the label when no fresh target exists (homing).
P4  Joint-space policy: predicts the joint target; at inference it replays
    through the same clamp/rate-limit/gripper cap.
P5  End-effector policy: predicts the constrained end-effector target in each
    arm's own base frame; at inference it replays through the same inverse
    kinematics and envelope projection that produced the label. Available only
    when the Cartesian interface (and its solver) runs.
P6  Frozen action definition: the constants that define the action — units,
    joint order, gripper cap, scales, frame convention, quaternion order — are
    stored with the dataset so training and inference agree exactly; the
    end-effector features are stored under neutral keys so a default joint
    policy is unaffected by their presence.

P-new-1  When there is no slot that means the same thing: the mapping above
    presumes a correspondence, and a tactile platform breaks it twice — no
    pretrained slot has seen a deformable-contact image, and we record more
    cameras than there are slots. A known viewpoint keeps its slot by name; an
    unknown one takes the next free slot, stated as arbitrary placement rather
    than correspondence; more cameras than slots is REFUSED, not silently
    dropped, because a run that drops an input reports as a result about
    inputs it never had. The assignment is per run, so it is an experimental
    factor rather than a property of the code.
P-new-2  Fitting more views than a policy accepts: some policies concatenate
    their views into one fixed frame, so the count is exact, not bounded.
    Rather than discard three of four fingertips we COMPOSE — tile the four
    tactile streams into one image feature, frame for frame, occupying one
    view beside the exterior camera. Resolution is traded for coverage; the
    tiling lives in the dataset presented to the learner, so it is visible and
    reproducible. Unjustified: composed vs chosen-subset is not measured.
P-new-3  Where the training runs: the accelerators are elsewhere, as the
    forward pass is. A destination is configuration — how work is queued,
    where scratch lies, what the card was MEASURED to carry — and everything
    after choosing a run is identical wherever it lands. The point is early
    refusal: a missing camera, an impossible view count or a batch over a
    measured ceiling is reported in seconds, not hours into a reservation. A
    machine with no measured ceiling refuses nothing, which beats a plausible
    number that would be wrong in both directions.
