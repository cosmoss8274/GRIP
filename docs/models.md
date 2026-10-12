# Model interfaces

GRIP uses a shared sequence encoder and energy backbone. The classifier adds a trained head to that backbone; the inverse model uses structural features through CarbonDesign.

| Component | Input | Output |
| --- | --- | --- |
| `BaseEnergyModel` | Peptide and receptor sequences | Single-site, pair and total energies |
| `ClassifierModel` | Peptide and receptor sequences | Classification logit and sigmoid score |
| Inverse interface | Prepared peptide–MHC–TCR structure features | Residue and residue-pair logits |

The sequence model depth is set by `config.model.num_layers`. Use the configuration associated with the checkpoint; all weights must match it exactly. There is one backbone implementation for both energy evaluation and classification.

## Sequence energy

```python
from grip_models import load_pretrained, score_record

model, config = load_pretrained("mhc-i-energy")
energies = score_record(model, record, config)
```

The released weights provide the MHC-I base model. `single` sums the negative log probabilities of the observed residues. `pair` sums joint residue-pair negative log probabilities over the ordered site matrix, including its diagonal, with a factor of one half. `total` is their sum. These outputs retain the reference energy scale.

See [Usage](usage.md) for the record format and local checkpoint loading.

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

Provide a complete fine-tuned checkpoint, including its classification head. `checkpoint_digest` identifies that local file; use the digest supplied with your checkpoint. Classification weights are not part of the existing MHC-I energy download.

The `mhc` calculation divides the single-site energy by peptide length and the pair energy by length squared. These three normalized energy values, 21 mean sequence logits and 441 mean pair logits form the 465-feature classifier input. The `tcr_legacy` calculation preserves the two successive normalizations used by the original TCR evaluation. Select the profile associated with the checkpoint and evaluation procedure.

`classify_record` evaluates one record at a time and returns `logit` and `score`. The score is `sigmoid(logit)`; it is not an experimentally calibrated binding probability. A binary decision requires a separately validated threshold.

The receptor representation must match the evaluation procedure. The historical MHC-I benchmark encoded allele-label strings directly, whereas the experimental-screen evaluation used MHC pseudosequences. Those are different inputs. This sequence API accepts canonical amino-acid strings; it does not silently convert HLA names or reproduce the historical label-string encoding. The original MHC-I training-input representation remains unconfirmed.

## Structure-conditioned inverse model

The inverse interface loads the complete GRIP checkpoint into an independently initialized CarbonDesign backbone and returns logits on the original full residue axes. ESM2 representations, structural feature definitions and recycling settings follow the original configuration. See [inverse-model usage](inverse.md) for dependencies and the feature interface.

These interfaces provide the neural components. Peptide sampling, physical refinement and task-specific experimental decision rules are separate from model inference.
