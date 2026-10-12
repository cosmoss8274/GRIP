# Inverse model

The inverse model predicts residue and residue-pair sequence preferences in a
peptide–MHC–TCR complex for structure-conditioned peptide design.

## Usage

Complete the [dependency setup](dependencies.md#inverse-model) to initialize
`backbone`, then load a local GRIP inverse checkpoint:

```python
import torch
from grip_models import load_inverse_model, inverse_logits

model = load_inverse_model(
    backbone,
    "grip-inverse-state.pt",
    sha256=checkpoint_digest,
    device="cpu",
)
features = torch.load("prepared_structure_features.pt", weights_only=True,
                      map_location="cpu")
output = inverse_logits(model, features)
```

`checkpoint_digest` is the supplied file checksum. For weights stored in a
training checkpoint, pass the appropriate `state_dict_key`.

## Structure features

Prepare a fresh batch with the structure feature pipeline and its
[feature configuration](https://github.com/zhanghaicang/carbonmatrix_public/blob/main/config/config_data_mrf2.json).
Use the following tensors on the model's device, where `B` is batch size and
`L` is the number of residues in the complete complex.

| Feature | Shape | Type |
| --- | --- | --- |
| `seq` | `[B, L]` | int64 |
| `mask`, `peptide_mask` | `[B, L]` | bool |
| `chain_id` | `[B, L]` | int32 or int64 |
| `chain_types` | `[B, L]` | int64 |
| `dist_one_hot` | `[B, L, L, 500]` | float32 |
| `pair_mask` | `[B, L, L]` | bool |
| `left_gt_calpha3_frame_positions`, `right_gt_calpha3_frame_positions` | `[B, L, 3]` | float32 |
| `rigidgroups_gt_frames` | rotations `[B, L, 8, 3, 3]`, translations `[B, L, 8, 3]` | float32 |

The feature pipeline sets `chain_id` to zero and encodes `chain_types` as
0 (MHC), 1 (TCR α), 2 (TCR β) and 3 (peptide). Preserve its atom ordering and masks.

## Outputs

| Key | Shape | Meaning |
| --- | --- | --- |
| `logits` | `[B, L, 21]` | Residue sequence preferences |
| `pair_logits` | `[B, L, L, 441]` | Joint residue-pair preferences |
| `peptide_mask` | `[B, L]` | Peptide positions |

Outputs span the complete complex. The amino-acid order is
`ARNDCQEGHILKMFPSTWYVX`; the pair axis follows its 21 × 21 combinations.
Sequence decoding is a separate step.
