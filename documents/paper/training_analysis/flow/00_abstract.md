# Abstract — paragraph flow

Single paragraph, high level, NO numbers (writing guideline §4).

- Hook: a tactile-equipped bimanual rig learns garment folding from
  teleoperated demonstrations, and the question asked of it was whether the
  policies read the tactile sensors or merely the light leaking round their
  edges.
- Problem: the evidence that prompted the question was gradient attribution on
  training recordings, and neither attribution on training data nor a training
  loss can answer whether a change helps.
- What we do: retrain every policy with a held-out split, score them on
  recordings they never saw, and compare cropped tactile inputs against
  uncropped ones under matched budgets.
- Turn: report that the comparison reverses several conclusions drawn from
  training data, and that two of them reverse again once the policies are
  compared over a horizon they share.
- Close: state that the report gives every figure's provenance so each claim
  can be checked, and flags the confounds that remain open.
