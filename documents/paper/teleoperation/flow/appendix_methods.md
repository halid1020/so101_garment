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
Final subsection: common input processing (kept, extended with the
engagement behaviour cross-reference to the method section)
```
