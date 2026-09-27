# What the policies do, trained and unseen — paragraph flow

The section that carries the headline table. Every claim in it must be readable
off a table or figure in the same section.

- P1 opens, says what is compared and forward-links the table and two figures.
- P2 THE HORIZON PROBLEM, stated before any number is quoted, because every
  number depends on it: the families plan different distances ahead, so an
  error averaged over each family's own chunk compares unequal questions. Say
  what is done about it -- report both the native figure and one over the
  shared horizon -- and that the shared one is what cross-family claims rest on.
- P3 the table, read out: which family generalises better on unseen data, and
  by how much on the shared horizon rather than the native one.
- P4 the gap between trained-on and unseen error, per family. This is the
  largest effect in the report and it is within a family, so the horizon
  problem does not touch it. Name what it does and does not establish, and
  forward-link the drift confound.
- P5 the shape of the decay across a chunk, which the single numbers hide: one
  family is flat across its whole chunk on recordings it trained on and steep
  on recordings it did not. Say what that shape means -- memorisation -- rather
  than leaving the reader to infer it.
- P6 (added 2026-09-25, when the third family's rows landed) the fine-tuned
  family, read with its training budget beside it. Its gap is the smallest in
  the table, and that is NOT generalisation: it was trained for about one pass
  over the recordings and its error on recordings it trained on is already
  higher than the other two families' error on recordings they never saw. A
  ratio near one here means it never fitted the training half. Say so before a
  reader credits it. Its crop difference is a few per cent with no sampler floor
  measured, so no claim is made from it. It reads three cameras, not five.
