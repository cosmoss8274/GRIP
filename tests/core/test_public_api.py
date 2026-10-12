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

    def test_import_and_discovery_do_not_import_model_or_dependencies(self):
        self.run_isolated("""
            import importlib.abc

            class RejectModelImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    blocked = {'torch', 'numpy', 'pandas', 'scipy', 'sklearn',
                               'einops', 'Bio', 'grip_models.sequence'}
                    if any(fullname == name or fullname.startswith(name + '.')
                           for name in blocked):
                        raise AssertionError('Unexpected model import: ' + fullname)

            sys.meta_path.insert(0, RejectModelImports())
            import grip_models
            assert grip_models.__all__ == [
                'BaseEnergyModel', 'TrainingDataGenerator', 'load_config',
                'prepare_record', 'score_record', 'load_weights']
            assert set(grip_models.__all__).issubset(dir(grip_models))
            assert 'grip_models.sequence' not in sys.modules
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
            stub = ModuleType('grip_models.sequence')
            stub.myModel = object()
            stub.DataGenerator = object()
            sys.modules[stub.__name__] = stub
            inference = ModuleType('grip_models.inference')
            for name in ('load_config', 'prepare_record', 'score_record'):
                setattr(inference, name, object())
            sys.modules[inference.__name__] = inference
            checkpoints = ModuleType('grip_models.checkpoints')
            checkpoints.load_weights = object()
            sys.modules[checkpoints.__name__] = checkpoints
            assert grip_models.BaseEnergyModel is stub.myModel
            assert grip_models.TrainingDataGenerator is stub.DataGenerator
            assert grip_models.load_config is inference.load_config
            assert grip_models.prepare_record is inference.prepare_record
            assert grip_models.score_record is inference.score_record
            assert grip_models.load_weights is checkpoints.load_weights
            assert grip_models.__dict__['BaseEnergyModel'] is stub.myModel
        """)


if __name__ == "__main__":
    unittest.main()
