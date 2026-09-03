#!/usr/bin/env bash
set -u
cd ~/project/so101_garment
export MUJOCO_GL=egl PYTHONPATH=.:src
RUN=outputs/vla_sim_long/handover_split_200
OUT=$RUN/eval/val
mkdir -p $OUT
for pol in act diffusion; do
  for ck in $RUN/full/handover_split/$pol/checkpoints/0*; do
    step=$(basename $ck)
    [ -f "$OUT/${pol}_${step}.json" ] && { echo "skip $pol $step"; continue; }
    echo "===== $pol $step  $(date -Is)"
    venv/bin/python -u tool/eval_sim_policy.py \
      --task handover_split --seeds val --fps 25 --device cuda \
      --camera-width 640 --camera-height 480 \
      --checkpoint $ck/pretrained_model \
      --out $OUT/${pol}_${step}.json 2>&1 | tail -3
  done
done
echo "CKPT SWEEP DONE $(date -Is)"
