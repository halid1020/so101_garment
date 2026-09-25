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

## Action error subsection (added 2026-09-25, from the results)

- A1 the horizon problem again, sharper: the large model plans only ten steps,
  so ten is the one horizon every model shares; everything in this table is
  rescored over it from the per-step curves already computed.
- A2 the table, read out, in bold: both world models plan the next third of a
  second better on unseen recordings than any policy -- about a fifth below the
  best policy -- and within a few per cent of each other, which is too close to
  rank.
- A3 the small model's shape: it fits its training recordings more tightly than
  anything in the report and has the widest gap, so it also memorises most.
- A4 what this does not show, forward to threats: one short horizon, no sampler
  floor for either world model, the large model sampled less densely, and the
  small model sees 0.8 s of past video where the policies see one frame -- so
  the advantage is not attributable to the prediction objective alone.
- A5 (added 2026-09-25, once measured) the sampler floor, as its own small
  table: three seeds over the held-out recordings. The small world model's lead
  over the best policy is many times its own spread, so A2 stands for it; the
  fine-tuned policy's near-term error swings a fifth between seeds and it is
  last on every seed. The large model's floor is pending and A2 says so.
