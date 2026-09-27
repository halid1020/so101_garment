# 2 Methods

P0  intro: models, then data and split, then evaluation, then the crop
2.1 Models
P1  shared: proprioception, chunks at 30 Hz; the rate checked in the recordings
    (frame age median 16 ms, p90 30 ms -> 30 Hz cameras, not 25)
P2  one paragraph per model, original -> ours, each saying how the cameras are
    encoded (shared vs per-camera ResNet, tokens, tiling, composite) and how
    much history it sees: ACT, Diffusion, flow matching, pi0.5 (authors
    finetune fully; LoRA literature: OpenVLA r=32 all layers ~ full,
    Biderman: learns less; ours r=16 action expert only), DreamZero (authors
    finetune the 14B model on ~30 min; ours 59M from scratch; tiles 5 cameras
    into a 3x3 grid; also given 1.6 s of past commands), FastWAM (Wan2.2,
    authors finetune per task; overhead | 2x2 composite; skips video at test)
T1  models: type, plan, fingertips, starts from, parameters, history
F1  the images each model receives
2.2 The demonstrations and the split
P3  one session = 65 demonstrations in one afternoon (~1 h 45 min); fixed start
F2  20 frames of one test demonstration
P4  last seven are the latest; drift would make them harder for reasons other
    than generalisation; random seven sit between training demos in time, so
    drift would make them easier -> the control
2.3 Evaluation
P5  action error, with the sample counts explained (every 25th frame, ~18 per
    demonstration, 1127 / 111); first ten steps for comparison
P6  Grad-CAM (ResNet models only; why not pi0.5 or the WAMs)
P7  patch occlusion maps: the same question for every model
P8  input contribution
P9  world action models: reconstruction vs prediction; PSNR (what dB means),
    SSIM, both higher is better; the repeat-last-frame baseline and margin
2.4 The tactile crop
F3  the Grad-CAM picture that raised the question, with edge ratios
P10 the images before contact: edges 10-21 % brighter, but the ridge at ~0.3
    across is the brightest feature (up to 32 %)
F4  rim evidence with the ridge marked and all crop boxes
P11 four crops: rows only (and why not columns), rows unstretched (ACT),
    four edges, ridge crop (rows 0.1-0.9, columns 0.36-0.9); in the saved
    pipeline, so applied at test time too
