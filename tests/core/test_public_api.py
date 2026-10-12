"""Check lazy package behavior without importing or executing the model source."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class PublicApiTests(unittest.TestCase):
    def run_isolated(self, code):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             "import sys\nsys.path.insert(0, " + repr(str(SRC)) + ")\n"
             + textwrap.dedent(code)],
            capture_output=True, text=True, env=environment, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_discovery_and_config_do_not_import_model_dependencies(self):
        self.run_isolated("""
            import importlib.abc

            class RejectModelImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    blocked = {'torch', 'numpy', 'pandas', 'scipy', 'sklearn',
                               'einops', 'Bio', 'grip_models.model', 'grip_models.data'}
                    if any(fullname == name or fullname.startswith(name + '.')
                           for name in blocked):
                        raise AssertionError('Unexpected model import: ' + fullname)

            sys.meta_path.insert(0, RejectModelImports())
            import grip_models
            assert grip_models.__all__ == [
                'BaseEnergyModel', 'ClassifierModel',
                'SequenceDataset', 'TrainingDataGenerator', 'load_config',
                'prepare_record', 'score_record', 'load_weights', 'load_pretrained',
                'load_classifier', 'classify_record', 'load_inverse_model',
                'inverse_logits']
            assert set(grip_models.__all__).issubset(dir(grip_models))
            assert 'grip_models.model.energy' not in sys.modules
            assert callable(grip_models.load_pretrained)
            config = grip_models.load_config()
            assert config.model.embeddings.seq_channel == 128
            assert config.model.embeddings.pair_channel == 64
            try:
                grip_models.unknown_export
            except AttributeError:
                pass
            else:
                raise AssertionError('Unknown export must raise AttributeError')
        """)

    def test_aliases_use_original_symbols_with_a_stub_module(self):
        self.run_isolated("""
            from types import ModuleType
            import grip_models

            # Deliberately use a stub: this is not a model runtime test.
            stub = ModuleType('grip_models.model.energy')
            stub.BaseEnergyModel = object()
            sys.modules[stub.__name__] = stub
            data = ModuleType('grip_models.data')
            data.SequenceDataset = object()
            sys.modules[data.__name__] = data
            inference = ModuleType('grip_models.inference')
            for name in ('load_config', 'prepare_record', 'score_record'):
                setattr(inference, name, object())
            sys.modules[inference.__name__] = inference
            checkpoints = ModuleType('grip_models.checkpoints')
            checkpoints.load_weights = object()
            sys.modules[checkpoints.__name__] = checkpoints
            pretrained = ModuleType('grip_models.pretrained')
            pretrained.load_pretrained = object()
            sys.modules[pretrained.__name__] = pretrained
            assert grip_models.BaseEnergyModel is stub.BaseEnergyModel
            assert grip_models.TrainingDataGenerator is data.SequenceDataset
            assert grip_models.load_config is inference.load_config
            assert grip_models.prepare_record is inference.prepare_record
            assert grip_models.score_record is inference.score_record
            assert grip_models.load_weights is checkpoints.load_weights
            assert grip_models.load_pretrained is pretrained.load_pretrained
            assert grip_models.__dict__['BaseEnergyModel'] is stub.BaseEnergyModel
        """)


if __name__ == "__main__":
    unittest.main()
