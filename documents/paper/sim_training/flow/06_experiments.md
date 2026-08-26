# 6 Experiments — paragraph flow

1. Intro paragraph: the experiment asks whether three policy families
   (ACT, Diffusion Policy, pi0.5) learn the two tasks from oracle
   demonstrations, and how the simple and full environment modes
   separate bugs from data problems. Forward links to protocol and
   (pending) results.
2. Protocol: oracle gate → gated collection (~one demonstration per
   training seed) → training per policy → every checkpoint validated on
   the validation seeds → the selected checkpoint evaluated once per
   evaluation seed with videos. Wall-clock notes.
3. Policies and sizing: ACT and Diffusion from scratch; pi0.5 adapted
   from its published base with low-rank adapters (full finetuning does
   not fit the target GPU).
4. Results: TODO — tables and per-seed heat-maps land here from the
   long-run report and analysis notebook once the full run completes on
   the training machine. (Stub with the table skeleton.)

Additional paragraph — what simple mode does and does not measure:
state, from a direct measurement of the collected dataset, that every
demonstration in simple mode begins from an identical state: the spread
of the first recorded state across episodes is zero on every channel.
Simple mode also draws its validation and evaluation scenarios from the
same seed as its training data, so a simple-mode success rate reports
how completely a policy has fitted one scene and says nothing about
generalisation. Full mode, and the split-reach relay in particular,
carry the initial-state variation the question actually needs.
