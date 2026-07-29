# 1 Introduction — paragraph flow

1. Motivation: imitation-learning pipelines for bimanual manipulation
   have many failure points (data schema, normalisation, checkpointing,
   evaluation) that surface only after expensive demonstration
   collection; proving the loop in simulation first de-risks the real
   campaign. Hands over to: what exists already.
2. Context: the companion teleoperation paper describes the dual 5-DoF
   rig and its Quest teleoperation stack; the rig has a faithful digital
   twin generated from the same geometry source as the physical build.
   Hands over to: the gap.
3. Gap: a twin alone is not a training environment — it needs graspable
   objects with believable contacts, tasks with measurable success,
   demonstrations whose distribution matches teleoperated data, and an
   evaluation protocol that cannot leak training scenarios.
4. Contributions list (environment, tasks + seed protocol, pipeline-
   authentic oracle + closed-loop corrections, difficulties catalogue,
   experiment harness). Forward links to each section.
