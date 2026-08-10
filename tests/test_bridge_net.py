import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import simple_nn


class IdeaBridgeNetTests(unittest.TestCase):
    def test_bridge_net_builds_and_returns_bridge_state(self):
        model = simple_nn.IdeaBridgeNet(input_dim=7, branch_hidden=(8, 6), bridge_dim=4, output_dim=3)
        left = torch.randn(2, 7)
        right = torch.randn(2, 7)

        output, trace = model(left, right)

        self.assertEqual(output.shape, (2, 3))
        self.assertIn("left", trace)
        self.assertIn("right", trace)
        self.assertIn("bridge", trace)
        self.assertIn("fused", trace)
        self.assertEqual(trace["bridge"].shape, (2, 4))
        self.assertEqual(trace["fused"].shape, (2, 4))

    def test_pair_targets_still_capture_bridge_signal(self):
        left = [1, 0, 1, 1, 0, 1, 0]
        right = [1, 0, 1, 1, 0, 1, 0]
        targets = simple_nn.make_pair_targets(left, right)

        self.assertEqual(targets[0], 1)
        self.assertEqual(targets[1], 1)
        self.assertEqual(targets[2], 1)

    def test_cat_car_demo_data_is_well_formed(self):
        left_data, right_data, targets = simple_nn.make_cat_car_demo_data()

        self.assertEqual(len(left_data), len(right_data))
        self.assertEqual(len(left_data), len(targets))
        self.assertEqual(len(left_data[0]), 7)
        self.assertEqual(len(targets[0]), 3)

    def test_tri_bridge_net_builds_and_returns_bridge_state(self):
        model = simple_nn.IdeaTriBridgeNet(input_dim=7, branch_hidden=(8, 6), bridge_dim=4, output_dim=3)
        left = torch.randn(2, 7)
        center = torch.randn(2, 7)
        right = torch.randn(2, 7)

        output, trace = model(left, center, right)

        self.assertEqual(output.shape, (2, 3))
        self.assertIn("left", trace)
        self.assertIn("center", trace)
        self.assertIn("right", trace)
        self.assertIn("bridge", trace)
        self.assertIn("fused", trace)
        self.assertEqual(trace["bridge"].shape, (2, 4))
        self.assertEqual(trace["fused"].shape, (2, 4))

    def test_triple_demo_data_is_well_formed(self):
        data = simple_nn.make_triple_bridge_demo_data()

        self.assertEqual(len(data[0]), len(data[1]))
        self.assertEqual(len(data[0]), len(data[2]))
        self.assertEqual(len(data[0][0]), 7)
        self.assertEqual(len(data[3][0]), 3)


if __name__ == "__main__":
    unittest.main()
