# The world models — paragraph flow

Appended to the same report (the user's choice), so it must read as a
continuation rather than a second paper. Gated on both models finishing.

- P1 opens the section and says what is new about these two: they predict future
  OBSERVATIONS as well as actions, which is a question no other arm in the
  report can be asked at all. Forward-links the two subsections.
- P2 what the two are and why they bracket the question: one is small and
  trained from scratch on this rig's data alone; the other is a large model
  carrying a pretrained video prior. State the parameter scale difference
  plainly. The pair is the experiment -- does a video prior buy anything at rig
  scale, or does the data do it?
- P3 that both are scored under the SAME treatment as the policies -- held-out
  recordings, sampler pinned, shared horizon where one exists -- so their action
  rows can be read against the earlier table rather than beside it.
- P4 hands over: action error first, because it is the column the rest of the
  report already established; prediction second, because it is the new thing.
