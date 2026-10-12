"""The configured energy-model depth executes every block and retains head axes."""

import unittest

import torch

from grip_models import BaseEnergyModel, load_config, prepare_record


class ModelDepthTests(unittest.TestCase):
    def test_four_and_sixteen_layer_models_execute_all_blocks(self):
        for depth in (4, 16):
            with self.subTest(depth=depth), torch.random.fork_rng(devices=[]):
                torch.manual_seed(712)
                config = load_config()
                if depth == 16:
                    config.model.num_layers = depth
                model = BaseEnergyModel(config).eval()
                self.assertEqual(len(model.evmodel.evos), depth)
                calls = []
                handles = [
                    block.register_forward_hook(
                        lambda module, args, output, index=index: calls.append(index)
                    )
                    for index, block in enumerate(model.evmodel.evos)
                ]
                batch = prepare_record(
                    {"epitope_sequence": "AAAAAAAA", "mhc": "ACDE"}, config
                )
                try:
                    with torch.inference_mode():
                        output = model(batch, compute_loss=False)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(calls, list(range(depth)))
                self.assertEqual(output["seqhead_pred"].shape, (1, 1, 8, 21))
                self.assertEqual(output["pairhead_pred"].shape, (1, 8, 8, 441))
                for value in output["energies"].values():
                    self.assertEqual(value.shape, (1,))
                    self.assertTrue(torch.isfinite(value).all())

    def test_different_depth_weights_cannot_load_strictly(self):
        four = BaseEnergyModel(load_config())
        config = load_config()
        config.model.num_layers = 16
        sixteen = BaseEnergyModel(config)
        with self.assertRaises(RuntimeError):
            sixteen.load_state_dict(four.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            four.load_state_dict(sixteen.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
