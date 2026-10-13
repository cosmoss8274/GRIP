# Sequence energy model

Use `load_pretrained("mhc-i-energy")` to load the MHC-I energy model and `score_record` to evaluate a peptide in its receptor context.

## Prepare a record

Save a peptide in `epitope_sequence` and its 34-residue MHC pseudosequence in `mhc`. The synthetic strings below illustrate the JSON format; replace them with the sequences to be scored:

```json
{
  "epitope_sequence": "AAAAAAAA",
  "mhc": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"
}
```

Use uppercase standard amino-acid letters without spaces. Each sequence must fit `data.max_length`, which defaults to 500.

For TCR checkpoints, supply the corresponding `l_v`, `l_j`, `h_v`, `cdr3_b` and `h_j` components. Each record requires at least one MHC or TCR component. Omitted fields, empty strings and `null` denote absent components.

## Load and score

```python
import json
from grip_models import load_pretrained, score_record

model, config = load_pretrained("mhc-i-energy")
with open("record.json") as stream:
    record = json.load(stream)
energies = score_record(model, record, config)
```

The model is returned in evaluation mode on the CPU. Select an available GPU with `device="cuda:0"`.

From a cloned repository, the same operation is available as:

```bash
python examples/core.py score --record record.json
```

The command prints a JSON object containing `single`, `pair` and `total`.

## Cache and offline use

The model and configuration are downloaded once from the [v1.0.0 release](https://github.com/cosmoss8274/GRIP/releases/tag/v1.0.0). The default cache is `$XDG_CACHE_HOME/grip/v1.0.0` or `~/.cache/grip/v1.0.0`. Set `cache_dir` to use another directory. Cached files are checked before loading.

After the files are cached, require offline loading with:

```python
model, config = load_pretrained(cache_dir="model-cache", local_files_only=True)
```

The command-line equivalents are `--cache-dir model-cache --local-files-only`.

## Try input encoding

```bash
python examples/core.py encode
python examples/core.py encode --record record.json
```

Without `--record`, the command uses a built-in synthetic record and reports component lengths and tensor count.

Encoding is also available in Python:

```python
from grip_models import load_config, prepare_record

config = load_config()
batch = prepare_record(record, config)
```

## Use a local checkpoint

Supply compatible base-model weights and the matching configuration:

```bash
python examples/core.py score --record record.json --checkpoint model.pt --config config.json
```

Omit `--config` to use the core defaults. `model.num_layers` can be set in the configuration to match the checkpoint's trunk depth. If weights are stored under a top-level key, add that exact key, for example `--state-dict-key model_state_dict`. Use `--device cuda:0` for a GPU. Run `python examples/core.py score --help` for all options.

`load_weights(model, path, sha256=..., state_dict_key=None)` loads a local state dictionary using its file checksum. The command-line tool computes this checksum automatically.

| Public name | Purpose |
| --- | --- |
| `load_pretrained(name="mhc-i-energy", ...)` | Load pretrained weights and their configuration |
| `load_config(path=None)` | Load core defaults with optional JSON overrides |
| `prepare_record(record, config)` | Encode one record as CPU tensors |
| `BaseEnergyModel(config)` | Construct the base energy model |
| `load_weights(model, path, sha256=..., state_dict_key=None)` | Load a matching local state dictionary |
| `score_record(model, record, config)` | Return one record's energy values |
| `SequenceDataset` | Lower-level sequence dataset |

`TrainingDataGenerator` remains an alias for `SequenceDataset` for existing scripts. Classification and structure-conditioned logits use the [model interfaces](models.md).

## Development

The CPU test environment uses Python 3.12 and `requirements-test.txt`:

```bash
python -m pip install -r requirements-test.txt
python -m pip install --no-deps .
python -B -m unittest discover -s tests/core -v
python -B -m unittest discover -s tests/runtime -v
python -B tools/check_numerical_parity.py
```
