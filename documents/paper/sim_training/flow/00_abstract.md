# Abstract — paragraph flow

One paragraph, high level, no numbers (guideline §4).

1. Problem: visuomotor policy pipelines fail late and expensively when
   their first test is real-robot data; we want the whole
   collect→train→evaluate loop proven before any hardware demonstration
   exists.
2. What we built: a simulation training environment inside the rig's
   digital twin — contact-based cube manipulation tasks, a scripted
   oracle that drives the *full production teleoperation pipeline*, and
   gated demonstration collection where only verified successes are
   saved.
3. Why the oracle design matters: demonstrations traverse the same
   filtering, clutching, retargeting and asynchronous solver as human
   teleoperation, so the data distribution matches what the real rig
   will produce.
4. Close: evaluation protocol with disjoint scenario pools; the
   difficulties section records every failure and fix so the environment
   is reproducible; results section grows with the living document.
