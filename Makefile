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

.PHONY: test test-unit test-integration test-system test-system-vla paper lint

test: test-unit test-integration

test-unit:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m unittest discover -s test/unit -t .

test-integration:
	PYTHONPATH=$(PYTHONPATH) MUJOCO_GL=egl $(PY) -m unittest discover -s test/integration -t .

test-system:
	bash test/system/smoke_test_pipeline.sh

test-system-vla:
	bash test/system/smoke_vla_sim.sh

paper:
	cd documents/paper/teleoperation && latexmk -pdf main.tex
	cd documents/paper/sim_training && latexmk -pdf main.tex

lint:
	venv/bin/pre-commit run --all-files
