# Models

| Model | Purpose |
| --- | --- |
| Sequence energy | Evaluate peptide sequences in an MHC or TCR context |
| GRIP-CLA | Score peptide–receptor interactions |
| Inverse model | Predict sequence preferences for structure-conditioned peptide design |

## Sequence energy

```python
from grip_models import load_pretrained, score_record

model, config = load_pretrained("mhc-i-energy")
energies = score_record(model, record, config)
```

Returns single-site (`single`), pairwise (`pair`) and total (`total`) energies.
See [Usage](usage.md) for input formatting and local checkpoints.

## Classification

Download a checkpoint and [classifier_weights.json](https://github.com/cosmoss8274/GRIP/releases/download/v0.2.0/classifier_weights.json) to the same directory:

| Checkpoint | Manifest key | Profile | MHC input |
| --- | --- | --- | --- |
| [MHC-II](https://github.com/cosmoss8274/GRIP/releases/download/v0.2.0/grip-mhc-ii-classifier.pt) | `mhc-ii-classifier` | `mhc` | 34-amino-acid MHC-II pseudosequence |
| [TCR](https://github.com/cosmoss8274/GRIP/releases/download/v0.2.0/grip-tcr-classifier.pt) | `tcr-classifier` | `tcr_legacy` | Full MHC protein sequence, or empty when absent |

```python
import json
from grip_models import load_classifier, classify_record

with open("classifier_weights.json") as stream:
    weights = json.load(stream)["models"]
checkpoint = weights["mhc-ii-classifier"]  # or "tcr-classifier"
model, config = load_classifier(
    checkpoint["filename"],
    sha256=checkpoint["sha256"],
    profile=checkpoint["profile"],
)
with open("record.json") as stream:
    record = json.load(stream)
prediction = classify_record(model, record, config)
```

Use the [record fields](usage.md#prepare-a-record) with uppercase amino-acid sequences.
For MHC-II, provide `epitope_sequence` and the 34-residue `mhc` pseudosequence; leave the TCR fields empty or omitted.
For TCR, provide `epitope_sequence` and `cdr3_b`, with `l_v`, `l_j`, `h_v` and `h_j` when available. Its `mhc` input retains the full protein sequence (362 or 365 residues in the reference data), or is empty when unavailable. Allele names and MHC pseudosequences are not interchangeable with this input.

Each call evaluates one record and returns a classification `logit` and its
sigmoid `score`. Binary predictions require a task-validated threshold.

## Inverse model

The inverse model predicts amino-acid preferences within peptide–MHC–TCR
complexes for structure-conditioned peptide design. See [Inverse model](inverse.md)
for loading, structure features and outputs.
