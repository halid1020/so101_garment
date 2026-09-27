# How every figure was made — paragraph flow

Requested explicitly: plain, straightforward English. The writing guideline
forbids repository paths, class names and function names in prose, and the two
do not conflict -- describe what was computed and from what, and name no code.

- P1 opens and states the principle: every figure is regenerated from recorded
  result files, so no number is typed in by hand and a figure cannot drift away
  from the run it describes.
- P2 the action-error figures and the table: what is sampled, what is compared
  against what, how the per-step curve is formed, and how the shared-horizon
  column is derived from the same curve.
- P3 the contribution figures: how a share is produced, what the baseline
  substitution is, and how the frames are sampled.
- P4 the gradient maps: what they are computed with respect to, why they are
  shown as deviation from uniform rather than as raw mass, and why the cropped
  arms are drawn from the cropped input.
- P5 the videos: which recording, how frames are sampled, what each panel
  shows, and where the files are. Say that the still strips in this report are
  sampled from those same videos.
- P6 the rim measure: normalised coordinates, the flat-map reference, and the
  band widths reported.
- (added 2026-09-25) world-model paragraphs: the action table (rescoring over
  the first ten steps, each model's own sampling rate, the video window the
  small model is given), the prediction numbers and margin figure (held-out
  recordings, one frame in thirty, pinned seed, margin over the held frame,
  time axis from each model's own spacing), and the filmstrips (the scored
  tensors themselves, not a second rollout).
