# Sequence energy model

Evaluate a peptide in its receptor context using a local model checkpoint and its matching configuration.

## Prepare a record

Save the peptide in `epitope_sequence` and its MHC sequence in `mhc`, using the receptor representation supported by the checkpoint. The synthetic strings below illustrate the JSON format; replace them with the sequences to be scored:

```json
{
  "epitope_sequence": "AAAAAAAA",
  "mhc": "ACDEFGHIK"
}
```

Use uppercase standard amino-acid letters without spaces. Each sequence must fit `data.max_length`, which defaults to 500.

For TCR checkpoints, supply the corresponding `l_v`, `l_j`, `h_v`, `cdr3_b` and `h_j` components. Each record requires at least one MHC or TCR component. Omitted fields, empty strings and `null` denote absent components.

## Load and score

From a cloned repository:

```bash
python examples/core.py score --record record.json --checkpoint model.pt --config config.json
```

The command prints a JSON object containing `single`, `pair` and `total` energies. Omit `--config` to use the core defaults. Set `model.num_layers` to match the checkpoint's trunk depth. For a state dictionary stored under a top-level key, add that exact key, for example `--state-dict-key model_state_dict`. Use `--device cuda:0` for a GPU.

In Python:

```python
import json
from grip_models import BaseEnergyModel, load_config, load_weights, score_record

config = load_config("config.json")
model = BaseEnergyModel(config)
load_weights(model, "model.pt", sha256=checkpoint_digest)
model.eval()

with open("record.json") as stream:
    record = json.load(stream)
energies = score_record(model, record, config)
```

`checkpoint_digest` is the expected file checksum. The command-line example computes the checksum of the supplied local file automatically. Both interfaces require a checkpoint that matches the model configuration and input representation.

## Try input encoding

```bash
python examples/core.py encode
python examples/core.py encode --record record.json
```

Without `--record`, the command uses a built-in synthetic record and reports component lengths and tensor count. Encoding is also available in Python:

```python
from grip_models import load_config, prepare_record

config = load_config()
batch = prepare_record(record, config)
```

## Python interface

| Public name | Purpose |
| --- | --- |
| `load_config(path=None)` | Load core defaults with optional JSON overrides |
| `prepare_record(record, config)` | Encode one record as CPU tensors |
| `BaseEnergyModel(config)` | Construct the base energy model |
| `load_weights(model, path, sha256=..., state_dict_key=None)` | Load a matching local state dictionary |
| `score_record(model, record, config)` | Return one record's energy values |
| `SequenceDataset` | Lower-level sequence dataset |

`TrainingDataGenerator` remains an alias for `SequenceDataset` for existing scripts. Classification and structure-conditioned logits use the [model interfaces](models.md).

For direct model calls, group records by the lengths of the peptide and every receptor component; `score_record` and `classify_record` evaluate one record at a time.

## Development

The CPU test environment uses Python 3.12 and `requirements-test.txt`:

```bash
python -m pip install -r requirements-test.txt
python -m pip install --no-deps .
python -B -m unittest discover -s tests/core -v
python -B -m unittest discover -s tests/runtime -v
python -B tools/check_numerical_parity.py
```
