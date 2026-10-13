# Sequence energy model

Evaluate a peptide in its receptor context using the trained [sequence energy weights](https://github.com/cosmoss8274/GRIP/releases/download/v1.0.1/grip-energy.pt) and [configuration](https://github.com/cosmoss8274/GRIP/releases/download/v1.0.1/grip-energy.json). The configuration specifies the 16-layer model.

## Prepare a record

Save the peptide in `epitope_sequence` and its receptor context in the corresponding sequence fields. For MHC-only scoring with the released weights, `mhc` contains the 34-residue MHC pseudosequence. The synthetic strings below illustrate the JSON format; replace them with the sequences to be scored:

```json
{
  "epitope_sequence": "AAAAAAAA",
  "mhc": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"
}
```

Use uppercase standard amino-acid letters without spaces. Each sequence must fit `data.max_length`, which defaults to 500.

For TCR-conditioned scoring, supply the amino-acid sequences of the TCR alpha-chain V and J regions in `l_v` and `l_j`, and the beta-chain V, CDR3 and J regions in `h_v`, `cdr3_b` and `h_j`. The training representation uses the associated MHC chain sequence when MHC accompanies TCR inputs. Supply sequences rather than allele or gene names. Each record requires at least one MHC or TCR component; omitted fields, empty strings and `null` denote absent components.

## Load and score

From a cloned repository:

```bash
python examples/core.py score --record record.json --checkpoint grip-energy.pt --config grip-energy.json
```

The command prints a JSON object containing `single`, `pair` and `total` energies. Lower energies indicate greater model compatibility when comparing peptides of the same length in a fixed receptor context. Use the supplied configuration with the released weights and add `--device cuda:0` for a GPU.

For other checkpoints, set `model.num_layers` to match the model depth. Omitting `--config` uses the four-layer core defaults. If the state dictionary is stored under a top-level key, add that key with `--state-dict-key`.

In Python:

```python
import json
from grip_models import BaseEnergyModel, load_config, load_weights, score_record

config = load_config("grip-energy.json")
model = BaseEnergyModel(config)
load_weights(model, "grip-energy.pt", sha256=checkpoint_digest)
model.eval()

with open("record.json") as stream:
    record = json.load(stream)
energies = score_record(model, record, config)
```

`checkpoint_digest` is the expected file checksum, provided with the [model downloads](https://github.com/cosmoss8274/GRIP/releases/tag/v1.0.1). The command-line example computes the checksum of the supplied local file automatically. Both interfaces require a checkpoint that matches the model configuration and input representation.

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
