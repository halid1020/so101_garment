# 1 Introduction — paragraph flow

P1  Intro paragraph: this section motivates policy-ready collection and
    lays out the paper (platform -> collection system -> contract ->
    replicability -> difficulties). Forward links.
P2  Hook: behaviour cloning learns a map from observation at a moment to the
    intent at that moment; the value of a demonstration depends entirely on
    those two being the same moment and the intent being the executed command.
    -> narrows to the real rig.
P3  Obstacle: our rig has many independent cameras on shared USB buses plus
    proprioception at a higher rate; grabbing each stream's latest value at a
    frame time silently skews them, and the operator's raw target is not what
    the arm followed after constraints. -> motivates the two ideas.
P4  Idea one: stamp on read, align every stream to one reference time per
    frame, and record the residual drift so alignment is measurable, not
    assumed. -> forward link to the collection section.
P5  Idea two: record the projected and constrained target (the executed
    command) in both joint and end-effector form, and require inference to
    replay a policy's output through the identical command path. -> forward
    link to the contract section.
P6  Contributions list -> maps to the section structure.
