"""Total against trained parameters, for a frozen part and for LoRA adapters.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_count_parameters
"""

import unittest

import torch

from tool.count_parameters import count


class CountTest(unittest.TestCase):
    def test_frozen_parts_count_in_total_only(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
        model[0].requires_grad_(False)
        result = count(model)
        self.assertEqual(result["total"], 9 + 4)
        self.assertEqual(result["trained"], 4)

    def test_lora_trains_only_its_adapters(self):
        model = torch.nn.Module()
        model.base = torch.nn.Linear(4, 4)
        model.lora_A = torch.nn.Linear(4, 1, bias=False)
        model.requires_grad_(False)  # a LoRA checkpoint loaded for inference
        result = count(model)
        self.assertTrue(result["lora"])
        self.assertEqual(result["trained"], 4)


if __name__ == "__main__":
    unittest.main()
