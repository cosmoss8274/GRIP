# Reproducibility

The development package preserves the reference GRIP base model and provides explicit interfaces around it. Source preservation, interface behavior and reproduction of a trained result are separate checks.

## Source and environment

The six inherited model modules retain their original numerical implementation. The reference `test_energy_nomean_newdata_fix_120.py` is packaged as `sequence.py`; five imports were changed to package-relative imports. `src/grip_models/source_changes.json` records the file mapping, hashes and exact import changes. The source tests reverse those changes and compare complete file contents and syntax trees.

`pyproject.toml` declares package dependencies. `requirements-test.txt` pins the seven direct dependencies used in the Python 3.12 CPU development environment, including PyTorch 2.6.0. It is a tested development configuration, not a complete transitive lockfile or the historical training environment. Preserve the Python version, platform, full dependency versions and relevant accelerator settings with each reproduced result.

## Configuration, inputs and weights

Use `grip_models.load_config(config_path)` for JSON configuration. It merges known, correctly typed fields into the reference defaults and rejects unknown fields, duplicate keys and non-finite numbers. `load_config(None)` returns the reference defaults; these defaults do not identify the configuration that produced an arbitrary checkpoint. Fixed dimensions required by the reference architecture cannot be overridden through this adapter.

`prepare_record(record, config)` uses the original `TrainingDataGenerator` and its collation function with `integrated_data_path=None`. It does not read an implicit training-data file. The required field is `epitope_sequence`; at least one of `mhc`, `l_v`, `l_j`, `h_v`, `cdr3_b` or `h_j` must also be nonempty. Omitted optional components and `None` become empty strings, with zero component lengths. Provided sequences must already use the standard 20 uppercase amino acids; whitespace, modifications and unsupported characters are rejected. The adapter applies `data.max_length`, or the reference fallback of 500, to every component.

`load_weights(model, checkpoint_path, *, sha256, state_dict_key=None)` requires one local file and an expected 64-character hexadecimal SHA-256 digest. The digest and deserializer read the same open file handle. Deserialization uses `weights_only=True` and `map_location="cpu"`, with no unsafe fallback. Select a wrapped state dictionary with an explicit top-level key; `None` means the checkpoint itself is the state dictionary. Prefixes are preserved. All keys, tensor types, shapes, dtypes and layouts are checked before `load_state_dict(strict=True)`; meta tensors are rejected. Loading preserves the model's training mode and returns the verified digest, absolute input path and selected key.

`score_record(model, record, config)` requires an already loaded model whose modules are all in evaluation mode, with existing sequence and pair heads and one shared, materialized device. It returns finite scalar `single`, `pair` and `total` energies. It does not select weights, change model mode, create heads or aggregate ensemble members. Callers must pair the model, configuration and checkpoint according to the training run's recorded artifacts. A matching hash establishes file identity, not that training provenance.

The preserved reference implementation prints diagnostic information, including input sequences. Keep these diagnostics local when working with private inputs.

## Run the development checks

Install the tested dependencies and package in a dedicated Python 3.12 environment, then run from the repository root:

```sh
python -m pip install -r requirements-test.txt
python -m pip install --no-deps .
python -B -m unittest discover -s tests/core -v
python -B -m unittest discover -s tests/source -v
python -B -m unittest discover -s tests/runtime -v
python -B tools/verify_source.py
```

The runtime tests require actual PyTorch and pandas. Checkpoint tests use tiny synthetic linear layers, and scoring-interface tests use a small protocol stub. Encoding tests compare every returned tensor with the original data generator. These tests do not require a trained biological model or a released checkpoint.

## Recorded numerical comparison

Run `python -B tools/check_numerical_parity.py` from the repository root to reconstruct and verify the reference source in a temporary directory and repeat the comparison. The [recorded report](validation/source-parity.json) includes the source hashes and tested environment.

Three CPU cases compared the original and packaged model implementations using synthetic inputs and identical model state under PyTorch 2.6.0. All 290 state tensors matched. In each case, all 28 batch tensors and all six output tensors matched, with maximum absolute difference **0.0**.

This comparison checks the numerical effect of source relocation under that configuration. It does not verify trained weights, an ensemble, biological prediction accuracy or performance on a published benchmark.

## Reproducing a trained result

A trained-result release needs a versioned source snapshot, the exact configuration and preprocessing contract, checkpoint files with expected hashes and explicit state-dictionary keys, the corresponding environment, and the evaluation inputs and procedure. Expected outputs or summary metrics must identify the artifacts that produced them. Any ensemble additionally needs its exact members and aggregation rule. Those trained artifacts and a validated ensemble specification are outside this development release.
