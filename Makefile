# so101_garment — task runner.
#
# Test tiers (fastest/cheapest first):
#   unit         fast, pure-python/pinocchio; no MuJoCo, no network.
#   integration  MuJoCo scenes (headless EGL render).
#   system       full train -> checkpoint -> eval pipeline; network + time.
#
# All targets use the in-repo venv (there is no system python) and set
# PYTHONPATH=.:src (plus MUJOCO_GL=egl where a render backend is needed).

PY := venv/bin/python
PYTHONPATH := .:src

.PHONY: test test-unit test-integration test-system test-system-vla test-system-vla-real test-port-parity paper lint

test: test-unit test-integration

test-unit:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m unittest discover -s test/unit -t .

test-integration:
	PYTHONPATH=$(PYTHONPATH) MUJOCO_GL=egl $(PY) -m unittest discover -s test/integration -t .

test-system:
	bash test/system/smoke_test_pipeline.sh

test-system-vla:
	bash test/system/smoke_vla_sim.sh

# Real-data plumbing smoke: trains a policy on a collected dataset and reloads
# the checkpoint. Needs the data drive mounted; pass the dataset via DATASET_ROOT.
#   make test-system-vla-real DATASET_ROOT=/media/hdd/so101/short_fold
test-system-vla-real:
	bash test/system/smoke_vla_real.sh --dataset-root "$(DATASET_ROOT)"

# Does a ported policy train like the LeRobot one it came from? A few hundred
# CPU steps of each, same seed, compared step for step. Skips with no dataset.
#   make test-port-parity DATASET_ROOT=/media/hdd/so101/cube-pnp-new
test-port-parity:
	bash test/system/test_port_parity.sh --dataset-root "$(DATASET_ROOT)"

paper:
	cd documents/paper/teleoperation && latexmk -pdf main.tex
	cd documents/paper/sim_training && latexmk -pdf main.tex
	cd documents/paper/real_training && latexmk -pdf main.tex

lint:
	venv/bin/pre-commit run --all-files
