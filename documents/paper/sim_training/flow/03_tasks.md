# 3 Tasks and evaluation protocol — paragraph flow

1. Intro paragraph: two tasks share one cube and one success notion;
   this section defines the tasks, the scenario generator, the seed
   protocol and the gating rule. Forward links.
2. Single-arm pick-and-place: cube and target sampled on the acting
   arm's side; sides alternate across scenarios; the idle arm holds its
   pose.
3. Bimanual relay: left arm picks on the left and lays the cube at the
   midline; right arm picks it there and places on the right target.
   Design evolution paragraph: the earlier mid-air hand-over failed
   (unconstrained yaw about the vertical during carry made the second
   grasp unreliable; two grippers on one small object clashed); the
   table-mediated relay decomposes the task into two proven single-arm
   grasps while remaining genuinely bimanual and sequential.
4. Split-reach relay: a third task, identical in motion to the relay but
   with the target pushed OUTSIDE the picking arm's reach, so the
   hand-off is forced by geometry rather than merely demonstrated. State
   plainly why the plain relay does not guarantee this: its generator
   checks only that the acting arm can reach each keypose, so a sampled
   target may sit inside both envelopes.
   Asymmetry paragraph: the same exclusion cannot be applied to the cube.
   Pushing the cube beyond the other arm's reach costs the demonstrator
   its grasp, because a small cube at that extension exceeds the
   open-loop grasp accuracy, whereas excluding the target costs nothing
   — a release only has to land inside the success radius, while a grasp
   has to close on the object. Placing is forgiving where picking is not.
5. Scenario generation: seeded sampling with an analytic reachability
   check; one scenario per seed. The reachability check is on position
   alone; report that the grasp attitude was measured across the whole
   sampled region and is not what limits it.
6. Seed protocol: disjoint train / validation / evaluation pools;
   *simple* mode (every phase on one fixed scenario — the
   overfit-one-scenario sanity check: a policy that cannot master one
   scenario has a bug, not a data problem) versus *full* mode.
7. Success criterion and demo gating: placed within tolerance, settled,
   grippers released; failed demonstrations are never written to the
   dataset — retries per seed, then the seed is skipped and recorded.
   Oracle gate thresholds before any training run.
