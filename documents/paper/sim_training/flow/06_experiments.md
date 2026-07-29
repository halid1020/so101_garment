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
