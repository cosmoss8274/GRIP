# Usage notes

## Inputs

`score_record(model, record, config)` scores one peptide in its receptor context. Supply `record` as a Python dictionary:

| Field | Value |
| --- | --- |
| `epitope_sequence` | Peptide sequence (required) |
| `mhc` | MHC sequence |
| `l_v`, `l_j`, `h_v`, `cdr3_b`, `h_j` | TCR sequence components in the training input format |

Include at least one MHC or TCR component. Sequences use the 20 standard uppercase amino-acid letters, without spaces. Omitted context fields, empty strings and `None` represent absent components. The maximum component length is set by `data.max_length` (default: 500).

The encoder uses the original training data generator. Results are returned as a dictionary with `single`, `pair` and `total` energy values.

## Model files

Use the JSON configuration and checkpoint from the same training run. In the README example, `checkpoint_sha256` is the checkpoint's expected SHA-256 checksum.

`load_weights` accepts a plain state dictionary. For a checkpoint containing a named state dictionary, pass its top-level key:

```python
load_weights(
    model, "model.pt",
    sha256=checkpoint_sha256,
    state_dict_key="state_dict",
)
```

Loading checks the file digest and requires the state dictionary to match the model, including both energy heads. PyTorch loads tensors on the CPU with `weights_only=True`. Move the model to the desired device and call `model.eval()` before scoring.

The original model prints diagnostic information, including input sequences, to the console.

## Tests

The CPU checks use Python 3.12 and the dependencies in `requirements-test.txt`:

```bash
python -m pip install -r requirements-test.txt
python -m pip install --no-deps .
python -B -m unittest discover -s tests/core -v
python -B -m unittest discover -s tests/source -v
python -B -m unittest discover -s tests/runtime -v
python -B tools/verify_source.py
python -B tools/check_numerical_parity.py
```

The numerical comparison uses synthetic inputs and identical model state to compare the packaged core with the original implementation.
