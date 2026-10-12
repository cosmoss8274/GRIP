# Structure-conditioned inverse model

The inverse model returns residue and residue-pair logits for a prepared
peptide–MHC–TCR structure. It uses CarbonDesign with the GRIP-trained weights and
ESM2 t33 650M representations. The interface preserves the original three
recycling passes and full residue ordering.

CarbonDesign is an external dependency. Obtain its source and model resources
separately from the [CarbonDesign repository](https://github.com/zhanghaicang/carbonmatrix_public).
Its implementation and pretrained resources are not included in this package.
From the GRIP checkout, install the optional dependencies with `pip install '.[inverse]'`
and make the external checkout importable, for example with
`PYTHONPATH=/path/to/carbonmatrix_public`.

## Load and evaluate

Use the upstream [model configuration](https://github.com/zhanghaicang/carbonmatrix_public/blob/main/config/config_model_mrf_pair_enable_esm_sc.json),
which matches the configuration used by GRIP. Keep its model settings unchanged
and set only the local ESM path. Use the local
`esm2_t33_650M_UR50D.pt` and its `esm2_t33_650M_UR50D-contact-regression.pt`
companion. The ESM model is initialized by CarbonDesign; it is not part of the
GRIP state dictionary. Supply a local GRIP inverse checkpoint; the current
weight download contains only the MHC-I energy model.

```python
from argparse import Namespace
import json
from pathlib import Path

import ml_collections
import torch
from carbondesign.model.carbondesign import CarbonDesign
from grip_models import load_inverse_model, inverse_logits

upstream = Path("/path/to/carbonmatrix_public")
config_path = upstream / "config/config_model_mrf_pair_enable_esm_sc.json"
config = json.loads(config_path.read_text())
config["model"]["embeddings_and_seqformer"]["lm"]["model_path"] = str(
    upstream / "params/esm2_t33_650M_UR50D.pt"
)
# The official ESM resource stores its configuration as an argparse.Namespace.
with torch.serialization.safe_globals([Namespace]):
    backbone = CarbonDesign(ml_collections.ConfigDict(config["model"]))

model = load_inverse_model(
    backbone,
    "/path/to/grip-inverse-state.pt",
    sha256="270647dcbbbabddf74866ac2b4e1ceac0a4a418776763539178a4b4ecbc550ea",
    device="cpu",
)
features = torch.load("prepared_structure_features.pt", weights_only=True,
                      map_location="cpu")
output = inverse_logits(model, features)
sequence_logits = output["logits"]             # [B, L, 21]
pair_logits = output["pair_logits"]            # [B, L, L, 441]
peptide_mask = output["peptide_mask"]           # [B, L]
```

The example digest identifies the tensor-only export of the inspected GRIP
inverse checkpoint. All 975 state entries, including the original chain-type
embedding, are loaded strictly. A safely loadable training checkpoint can be
selected with `state_dict_key="model_state_dict"`.

## Feature contract

This is a low-level feature interface. Prepare the batch with the original
structure parsing and feature pipeline, using the upstream
[feature configuration](https://github.com/zhanghaicang/carbonmatrix_public/blob/main/config/config_data_mrf2.json). A general
PDB/mmCIF input builder is not included in this interface.

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

Keep the original masking, atom ordering and chain identifiers. The reference
parser sets `chain_id` to zero and uses `chain_types` values 0 (MHC), 1 (TCR α),
2 (TCR β) and 3 (peptide). The wrapper retains those values without reassigning
chains. All feature tensors must be on the model's device. Pass fresh features,
without recycling state from a previous call.

Both output residue axes include the full input structure. The amino-acid axis
uses `ARNDCQEGHILKMFPSTWYVX`; the flattened pair axis can be viewed as
`[B, L, L, 21, 21]` in that order. Outputs are raw logits, not calibrated binding
probabilities. Selecting peptide residues is a separate indexing operation.

## Reference

Ren, Yu, Bu and Zhang. *Accurate and robust protein sequence design with
CarbonDesign*. Nature Machine Intelligence **6**, 536–547 (2024).
[doi:10.1038/s42256-024-00838-2](https://doi.org/10.1038/s42256-024-00838-2).
CarbonDesign and ESM remain subject to their respective upstream terms.
