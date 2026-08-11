# Abstract — paragraph flow

Single paragraph, high-level, NO numbers (guideline §5).

- Hook: a teleoperated demonstration is only useful for autonomy if the
  recorded observation and the recorded intent line up in time and if the
  action the policy will emit is exactly the quantity that was logged.
- Problem: multi-camera rigs with independent USB streams and asynchronous
  proprioception make that alignment easy to lose, and the recorded target is
  often the raw operator input rather than the constrained command the robot
  actually followed.
- What we do: describe a real-rig collection system that timestamps every
  stream on read, aligns them to a common per-frame reference, and logs the
  drift so alignment quality is auditable; and that records the projected,
  constrained target — the quantity a policy must reproduce — in both joint and
  end-effector form so either policy variant trains from the same episodes.
- Contract: state the teleoperation-to-autonomy principle — inference replays
  the policy output through the identical command path that produced the label.
- Close: the platform is measured and its constants are stored for
  reproducibility; a difficulties section records what went wrong and how it
  was resolved; an experiments section grows with the living document.
