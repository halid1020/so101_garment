# 3 Experiments

P0  intro: four experiments, in the order of the questions
3.1 Comparing the policies
P1  table: train / held-out / gap for all models, own horizon and first 10
P2  figure: error vs step for all models on one axis scale; finding: world
    models lowest near term
P3  finding: Diffusion very low at the first steps, then climbs steeply; ACT
    flat (trained) / steady (held-out)
P4  the gap and the split control (figure only; random seven is harder)
P5  pi0.5: under-trained at one pass; the longer run's result
P6  flow matching: same objective family as pi0.5 on Diffusion's vision
3.2 Cropping the tactile images (secondary)
P7  table: none / rows / rows-no-stretch / four edges per model; finding
P8  Grad-CAM rim measure: policies do not over-read the rim; figure
3.3 Which sensors ACT needs
P9  table/figure: central only, central + 2 tactile, all; finding
P10 input percentages and the framewise view (touch in bursts)
3.4 World models: reconstruction and prediction
P11 table per sensor: reconstruction, prediction, holding
P12 figure margin + filmstrips; findings (metric misses small moving parts)
