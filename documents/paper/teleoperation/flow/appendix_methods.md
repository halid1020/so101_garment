# Flow — Appendix: implementation details of all methods

Purpose (request: full implementation detail): after reading this
appendix the reader understands the complete implementation of every
teleoperation method in the repository — algorithm, update law, every
parameter with its value and justification (or an explicit
unjustified-choice flag), initialisation/reset behaviour, and failure
modes.

```
P0  common interface + rate limiter (kept, no class names)
Per method (armplane/pink_full, pink_relaxed, dls, mink, scipy_ls,
telegrip), a fixed template:
    - algorithm and update law (equations)
    - parameter table in prose: value + why (or ⚠ unjustified)
    - initialisation / reset behaviour (what happens at engagement)
    - failure modes / known limitations
NEW subsection: `mymethod` — decoupled thumbstick wrist interface.
    - algorithm: relaxed-Pink position solve + per-side joint-space
      thumbstick trims (deadzone + expo shaping, rate integration),
      three attitude modes (hold/incremental/soft), floor guard,
      release re-anchoring
    - parameters: deadzone 0.15, roll/flex rate 60/45 deg/s, expo 2.0,
      trim signs — all ⚠ unjustified (engineering guesses, untuned)
    - reset/engagement: shares the common engagement behaviour; the
      trim clutch latches the arm's joints while a stick is deflected
    - failure modes: signs unverified on hardware; loses direct
      hand-attitude correspondence in hold/incremental
    - the operator-selectable orientation-cost override lives here too:
      one scalar retunes the relaxed attitude task at runtime
Final subsection: common input processing (kept, extended with the
engagement behaviour cross-reference and the orientation-cost override)
Closing paragraph: parameter provenance — the commented configuration
files are canonical; the 171 cm operator derivation of the translation
scale, with its anthropometric links flagged as estimates
```
