# What cropping did — paragraph flow

- P1 opens, forward-links the figures, and restates the question in one line:
  does removing the gel rim change what the policy achieves?
- P2 the answer on unseen data, per family, with the effect sizes. Include the
  sign for each family and resist rounding a small effect into a story.
- P3 the finding that only the horizon view exposes: for one family the crop
  helps early in the chunk and hurts late, so a single averaged number reports
  whichever the horizon happened to favour. This is the clearest argument in
  the report for reporting a curve rather than a scalar.
- P4 the rim measure, which tests the original hypothesis directly rather than
  through task error: how much attribution mass falls in the outer band, always
  against what a flat map would give at the same resolution. Report that
  neither family over-reads the rim and that cropping moves both TOWARD it,
  which is the opposite of what the hypothesis predicts.
- P5 the honest limit of P4: map resolutions differ by family, so the
  cross-family comparison is only fair at the widest band, and the
  within-family comparison is fair everywhere. -> threats.
- (added 2026-09-26) Separating the rim from the stretch. A third ACT arm
  crops and does NOT resize, identical otherwise. Say why only this family:
  the diffusion implementation refuses cameras of different shapes, and the
  fine-tuned model would pad the shorter image rather than stretch it, which is
  a different change again. Read the table: the unstretched crop is lowest at
  every horizon, the stretched crop's long-horizon penalty vanishes without the
  stretch. In bold, bounded: consistent with the stretch -- not the rim removal
  -- being what cost the crop at long range; one training run per arm and
  training-seed variance unmeasured, so a direction, not a result.
