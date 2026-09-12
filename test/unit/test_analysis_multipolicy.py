"""A checkpoint with no weights is an error, not a log line.

The analysis package itself moved to actoris_harena and is tested there. What
stays here is tool/eval_sim_policy.py's weight loading, which is this rig's.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_analysis_multipolicy
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class WeightsMustActuallyLoadTest(unittest.TestCase):
    """A checkpoint with no weights is an error, not a log line.

    `from_pretrained` on a directory with no model.safetensors prints
    "Returning model without loading pretrained weights" and hands back a
    RANDOMLY INITIALISED model. Everything downstream then runs perfectly and
    reports numbers about noise. That is exactly what happened to a pi0.5
    attribution deck: it was produced and plotted, and was only caught because
    two runs of the same analysis disagreed, which a pinned analysis cannot.
    """

    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())

    def test_a_directory_with_no_weights_is_refused(self):
        from tool.eval_sim_policy import _load_weights

        with self.assertRaises(SystemExit) as caught:
            _load_weights(str(self.tmp), object(), object)
        message = str(caught.exception)
        self.assertIn("randomly initialised", message)

    def test_the_refusal_names_what_it_looked_for(self):
        from tool.eval_sim_policy import WEIGHT_FILES, _load_weights

        with self.assertRaises(SystemExit) as caught:
            _load_weights(str(self.tmp), object(), object)
        for name in WEIGHT_FILES:
            self.assertIn(name, str(caught.exception))

    def test_an_adapter_with_no_base_is_refused(self):
        from tool.eval_sim_policy import _load_weights

        (self.tmp / "adapter_config.json").write_text("{}")
        with self.assertRaises(SystemExit) as caught:
            _load_weights(str(self.tmp), object(), object)
        self.assertIn("names no base model", str(caught.exception))

    def test_the_base_is_found_in_either_file(self):
        import json

        from tool.eval_sim_policy import _adapter_base

        (self.tmp / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "from/adapter"})
        )
        self.assertEqual(_adapter_base(self.tmp), "from/adapter")

        other = Path(tempfile.mkdtemp())
        (other / "train_config.json").write_text(
            json.dumps({"policy": {"pretrained_path": "from/train-config"}})
        )
        self.assertEqual(_adapter_base(other), "from/train-config")

    def test_no_base_anywhere_reads_as_empty_not_as_a_crash(self):
        from tool.eval_sim_policy import _adapter_base

        self.assertEqual(_adapter_base(self.tmp), "")


if __name__ == "__main__":
    unittest.main()
