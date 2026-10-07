"""The driver's checkpoint pruning, and its weights-only mode for learning curves.

A learning curve scores every saved step after the run, so those steps need
their weights; the optimiser state is two thirds of a checkpoint and nothing
reads it once training has moved on. The default still removes whole
superseded checkpoints.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_prune_checkpoints
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[2] / "test/system/long_vla_real.sh"


def prune(run_dir: Path, keep: int, weights: bool) -> None:
    script = (
        "set -e\n"
        f'RUN_DIR="{run_dir}"; KEEP_CKPTS={keep}; KEEP_WEIGHTS={int(weights)}\n'
        f'eval "$(sed -n "/^prune_checkpoints()/,/^}}/p" {DRIVER})"\n'
        "prune_checkpoints act"
    )
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.run_dir, ignore_errors=True)
        self.ck = self.run_dir / "train/act/checkpoints"
        for step in ("005000", "010000", "015000", "020000"):
            for part in ("pretrained_model", "training_state"):
                (self.ck / step / part).mkdir(parents=True)
        (self.ck / "last").symlink_to("020000")

    def test_default_removes_whole_old_checkpoints(self):
        prune(self.run_dir, keep=2, weights=False)
        left = sorted(p.name for p in self.ck.iterdir() if p.name != "last")
        self.assertEqual(left, ["015000", "020000"])

    def test_weights_mode_keeps_every_model(self):
        prune(self.run_dir, keep=2, weights=True)
        for step in ("005000", "010000"):
            self.assertTrue((self.ck / step / "pretrained_model").is_dir())
            self.assertFalse((self.ck / step / "training_state").exists())
        for step in ("015000", "020000"):
            self.assertTrue((self.ck / step / "training_state").is_dir())


if __name__ == "__main__":
    unittest.main()
