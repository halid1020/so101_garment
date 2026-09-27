# 3 Experiments

3.1 Comparing the models
T2  action error, best in bold;  F5  error against time ahead, all models
P1  WAMs best over the first ten steps (DreamZero also sees past commands)
P2  flow matching the best behaviour-cloning policy early; half pi0.5's error
P3  Diffusion accurate early, grows fastest; ACT flat on training data
P4  every model worse on unseen; the random-seven control rules out drift
    (figure in the appendix)
P5  pi0.5's small gap is under-fitting; three passes barely move it
3.2 Cropping the tactile images
T3  test error under each crop, best per row in bold
P6  crops change almost nothing (pi0.5 and ACT four-edge; others pending)
F6  Grad-CAM edge weight;  F7  patch maps per model and crop
P7  no average over-weighting of the edge; does cropping move the focus
3.3 Which sensors the models use
T4  ACT on three camera sets
P8  overhead-only ACT generalises better; one run each
F8  input shares, all six models;  F9  touch over time
P9  proprioception dominates Diffusion, pi0.5 and FastWAM; touch in bursts
3.4 How well the world action models predict the cameras
T5  per sensor: reconstruction, prediction, repeat last, margins
P10 most error is prediction, not compression (dB in plain words)
P11 DreamZero beats repeat-last from 1.6 s; FastWAM sharp but frame-averaged
    scores barely reward it
F10 head-to-head filmstrip: seen frame and its reconstruction, predictions,
    differences, on one time axis
