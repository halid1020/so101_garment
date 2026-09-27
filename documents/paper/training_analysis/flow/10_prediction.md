# Future-observation prediction — paragraph flow

The section the world models exist for. Its whole job is to stop a reader
believing a high number means a good prediction.

- P1 opens and states the trap before any figure: on a camera that barely
  changes, repeating the last observed frame scores extremely well. A model
  reported without that baseline looks good for a reason that has nothing to do
  with prediction. Give the measured figure for holding on a near-static camera.
- P2 the measure that follows from P1: on how many steps of the horizon does
  the model BEAT holding, per camera. Say why per camera and not pooled -- four
  of five cameras here are gel images that barely move until contact, so a
  pooled number is dominated by the cameras where prediction is easiest and
  least useful.
- P3 the quantitative table: per camera, per model, the horizon steps won, with
  PSNR and SSIM behind rather than in front.
- P4 the qualitative figure: predicted against actual against held-last-frame,
  across the horizon, for one winning camera and one losing camera. The point of
  showing both is that the reader should be able to SEE what a losing camera
  looks like -- typically a plausible image that is simply the present.
- P5 what the comparison between the two models shows about the video prior,
  stated only as far as one dataset and one task can support.
- P6 the limits: the sampler is pinned so a number is reproducible, but both
  models sample and a different seed gives a different rollout; and prediction
  quality is not task success.
- (added 2026-09-25, from the results) the two models predict over DIFFERENT
  spans -- 0.8 to 4.8 s against about 0.13 to 1.1 s -- so P3's table compares
  step counts within a model only, P4's figure plots against seconds, and P5's
  comparison is the single shared instant (0.8 s), stated as one point.
- (added 2026-09-25) P4b: the large model's filmstrip, because its overhead
  margin is near zero while its frames are sharp and move correctly -- the
  averaged score cannot see a small moving part against a static garment. That
  is a finding about the metric, stated in bold, and it is why P4 exists.
