# 2 Methods

P0  intro: models, then data and split, then the measures, then the crop
2.1 The models
P1  one short paragraph per model: original version (paper) -> our version
    (what we changed and why): ACT, Diffusion Policy, pi0.5, flow matching,
    DreamZero, FastWAM
T1  table of differences: inputs (cameras), horizon, trained from scratch or
    pretrained, trainable parameters, steps/batch, predicts images?
2.2 Data and split
P2  one session, 65 recordings, 30 Hz; Figure: 20 frames of one recording
P3  the split in plain words: training on the first 58, testing on the last 7
    ("last seven"); the control repeats it with 7 picked at random ("random
    seven") to check that the gap is not the session drifting
2.3 How we measure
P4  action error: RMSE between planned and recorded commands, per step of the
    plan; common first-10 steps for cross-model comparison
P5  Grad-CAM, how it is computed (gradient of planned action wrt the last
    feature map -> channel weights -> weighted sum -> ReLU -> upsample)
P6  input contribution percentages: replace one input with its dataset mean,
    measure the change in the plan, normalise to 100 %
P7  world models: reconstruction (seen frames through own autoencoder) vs
    prediction (future frames, generated with the model's own actions),
    PSNR/SSIM per sensor, against "repeat the last frame seen"
2.4 The tactile crop
P8  what we saw (edge attention) and what we measured in the images (edges
    brighter before contact, but a brighter ridge inside the gel); Figure
P9  three crops: rows only (top/bottom tenth), rows without stretching,
    all four edges; resized back; applied at training AND test time
