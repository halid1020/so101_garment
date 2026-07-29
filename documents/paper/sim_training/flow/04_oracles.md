# 4 Oracle demonstrators — paragraph flow

1. Intro paragraph: two oracles share the task scripts and the gating;
   they differ in how end-effector targets become joint commands. The
   teleop-imitating oracle is the default; the direct oracle is the
   deterministic fallback. Forward links.
2. Why pipeline-authentic demonstrations: a policy trained on data that
   bypassed the teleoperation stack sees none of its filter lag, clutch
   re-anchoring or asynchronous solver dynamics; demonstrations that
   traverse the full stack match the distribution the real rig records.
3. The scripted operator: implements the same reader interface as the
   headset, holds both grips, squeezes triggers; derivation of the
   inverse retargeting map (place the virtual hand so the calibrated
   pipeline produces the desired end-effector target — the closed-form
   inversion of the calibration equation).
4. Closed-loop corrections (the operator's eyes): the differential IK
   carries an azimuth-dependent steady-state positional error that a
   small cube does not tolerate; a low-gain, clipped grasp-alignment
   servo centres the pinch before closing, and a hold servo steers the
   *object* onto the intended track while gripped; corrections freeze at
   jaw close and at set-down (dragging fix). Why low gain + clipping
   (pipeline lag → windup).
5. The direct oracle: script targets straight into the solver each tick;
   synchronous, deterministic, faster than real time; used for gates and
   as insurance.
6. Achieved reliability: direct oracle at ceiling on both tasks; teleop
   oracle at ceiling on the single task and high-but-imperfect on the
   relay; per-seed retries recover nearly all seeds; only verified
   successes enter datasets.
