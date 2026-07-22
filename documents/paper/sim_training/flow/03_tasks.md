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
4. Scenario generation: seeded sampling with an analytic reachability
   check; one scenario per seed.
5. Seed protocol: disjoint train / validation / evaluation pools;
   *simple* mode (every phase on one fixed scenario — the
   overfit-one-scenario sanity check: a policy that cannot master one
   scenario has a bug, not a data problem) versus *full* mode.
6. Success criterion and demo gating: placed within tolerance, settled,
   grippers released; failed demonstrations are never written to the
   dataset — retries per seed, then the seed is skipped and recorded.
   Oracle gate thresholds before any training run.
