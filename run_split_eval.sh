#!/usr/bin/env bash
set -u
cd ~/project/so101_garment
export MUJOCO_GL=egl PYTHONPATH=.:src
RUN=outputs/vla_sim_long/handover_split_200
OUT=$RUN/eval
mkdir -p $OUT
for pair in "act:080000" "diffusion:100000"; do
  pol=${pair%%:*}; step=${pair##*:}
  echo "===== $pol @ $step  $(date -Is)"
  venv/bin/python -u tool/eval_sim_policy.py \
    --task handover_split --seeds full --fps 25 --device cuda \
    --camera-width 640 --camera-height 480 \
    --checkpoint $RUN/full/handover_split/$pol/checkpoints/$step/pretrained_model \
    --out  $OUT/${pol}_${step}.json \
    --video-dir $OUT/videos/${pol}_${step} \
    --gif-dir   $OUT/gifs/${pol}_${step} 2>&1 | grep -v "^Training:"
  echo "===== $pol done rc=$? $(date -Is)"
done
echo "ALL DONE $(date -Is)"
