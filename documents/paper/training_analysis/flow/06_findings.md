# Findings — paragraph flow

Each finding is one short bold claim followed by the evidence that supports it
and the caveat that bounds it. Nothing appears here that is not readable off a
figure or table earlier in the report.

- P1 opens and says how to read the list: bold claim, evidence, bound.
- F1 on unseen recordings the diffusion family generalises better than the
  action-chunking one. Evidence: the shared-horizon column. Bound: the margin
  is far smaller than the native-horizon figure suggests, and the direction is
  what survives.
- F2 the generalisation gap differs sharply between families. Evidence: the
  train-against-unseen figure. Bound: the held-out episodes are the last of the
  session, so drift is not separated from overfitting.
- F3 cropping the tactile rim buys nothing on unseen recordings. Evidence: the
  crop rows against their twins. Bound: the crop also stretches the image, so
  this is a result about this crop and not about rim removal alone.
- F4 the crop's effect changes sign with horizon for one family. Evidence: the
  decay figure. Bound: a single averaged number cannot show this, which is the
  reason the curve is reported.
- F5 neither family over-reads the gel rim, and cropping moves attribution
  toward it. Evidence: the rim measure against a flat map. Bound: fair across
  families only at the widest band.
- F6 the tactile channels contribute little on average and much more around
  contact. Evidence: the contribution figures and the framewise video. Bound:
  shares are within-policy, so this is an ordering claim.
- F7 a policy can be flat across its whole chunk on recordings it trained on.
  Evidence: the decay figure, trained-on panel. Bound: this is a statement
  about memorisation, not about on-robot behaviour.
