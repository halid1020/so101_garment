# 2 Simulation environment — paragraph flow

1. Intro paragraph: the environment is the rig's digital twin plus a
   contact-manipulable payload; this section walks provenance → payload
   contacts → timing → sensing. Forward links.
2. Twin provenance: one parametric CAD description is the single source
   of truth for both the printed rig and the simulation model; the twin
   is regenerated from it, so simulation geometry cannot drift from the
   physical build. Camera poses come from the same source.
3. Payload and contact design: a small cube; why a cube (both grippers
   interact with the same object; a symmetric grasp tolerates yaw
   error); the anti-slip contact recipe (elliptic friction cone, high
   impratio, no-slip iterations, fine timestep), fingertip pads with
   contact priority, and a bounded gripper actuator force so a blocked
   jaw cannot explode the solver. Flag unjustified constants.
4. Timing design: control at 30 Hz = dataset rate; physics timestep
   chosen so one control tick is an exact integer number of substeps —
   evaluation ticks are therefore identical to training frames.
5. Sensing: three mounted RGB cameras (scene + two wrists) as the only
   policy views; the goal is communicated visually by a translucent
   target disc on the table; 12-D proprioceptive state (five joints +
   gripper per arm) in the same units and layout as the real recorder.
6. Gripper cap: commanded opening capped to a fraction of mechanical
   range for consistency with the real trigger mapping; open fraction
   semantics match real command semantics.
