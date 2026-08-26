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

---

## 6.x Executing a chunked policy over a network link — paragraph flow

Added when the deployment sweep landed. This subsection belongs here, not
in the teleoperation or real-platform papers: it measures policies this
environment trained, inside this environment, and its instrument is the
same evaluation harness the protocol above already defines.

1. Intro: the protocol above evaluates a policy with inference and
   environment in one process, which is not how the rig runs one. The
   inference machine is elsewhere, a policy answers with a chunk of
   future actions rather than one, and something must decide what
   happens to the actions already queued when the next chunk lands.
   Forward-links to the strategy taxonomy, the protocol and the results.
2. Why a chunk needs a join at all: the round trip is long compared with
   the control period, so the arms either wait for the reply or keep
   executing a plan made from a stale observation. Name the two families
   — blocking and aligning — and say that the choice is normally left
   implicit, which is what this experiment makes explicit.
3. The strategies, defined so a reader could reimplement them without
   our code: hold-for-the-reply; queue-behind; discard-and-replace;
   execute-a-fraction-then-request; cross-fade over a window; weighted
   average of old and new. State which parameters each one reads. Flag
   with the unjustified macro that the fractions, window lengths and
   weights are grid points, not derived values.
4. Protocol: which checkpoint, which task, which seed protocol and why
   the fixed scenario is the right one here (an overfit-one-scenario
   policy cannot be asked about scenes it never saw, and holding the
   scene fixed leaves the splice as the only thing varying). The rates,
   the link, and the fact that inference ran on a separate machine over
   a real network.
5. The measures and why each: success rate; time to completion for the
   successes; the seam ratio, defined as the size of the joint-space
   step at a chunk boundary against the typical step, so a value near
   one means the joins are invisible; the fraction of ticks with nothing
   to command; path length; and the observed round trip.
6. Results table and what it says. Lead with the separation that is not
   close, then the ordering within each family, then what did not
   separate at this sample size.
7. Real-time chunking, and why this table has no row for it: its
   smoothing happens inside the generative model on the inference
   machine, so it is only a distinct method when that machine performs
   it. Ours does not, and a client-side fallback would be exactly the
   discard-and-replace row under a name that would mislead. State this
   as a scope boundary, not a result.
8. Threats to validity: one scenario, one policy family, one link;
   the sample size per cell; and that a blocking strategy's cost is paid
   in wall-clock time, which a simulator makes cheap and a real rig does
   not.
