# Models

| Model | Purpose |
| --- | --- |
| Sequence energy | Evaluate peptide sequences in an MHC or TCR context |
| GRIP-CLA | Score peptide–receptor interactions |
| Inverse model | Predict sequence preferences for structure-conditioned peptide design |

## Sequence energy

Load a local checkpoint with `BaseEnergyModel` and `load_weights`, then evaluate a record with `score_record`. The outputs are single-site (`single`), pairwise (`pair`) and total (`total`) energies.

See [Usage](usage.md) for configuration, input formatting and examples.

## Classification

```python
from grip_models import load_classifier, classify_record

model, config = load_classifier(
    "classifier.pt",
    sha256=checkpoint_digest,
    profile="mhc",
)
prediction = classify_record(model, record, config)
```

Use a task-specific checkpoint and its matching receptor sequences and preprocessing.
Set `profile="mhc"` for MHC-I/II checkpoints or `profile="tcr_legacy"` for TCR checkpoints.
`checkpoint_digest` is the supplied file checksum.

Each call evaluates one record and returns a classification `logit` and its
sigmoid `score`. Binary predictions require a task-validated threshold.

## Inverse model

The inverse model predicts amino-acid preferences within peptide–MHC–TCR
complexes for structure-conditioned peptide design. See [Inverse model](inverse.md)
for loading, structure features and outputs.
