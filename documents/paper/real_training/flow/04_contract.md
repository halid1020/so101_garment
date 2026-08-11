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
