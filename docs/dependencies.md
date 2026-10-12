# Dependencies

The standard installation includes the sequence energy model and classifier dependencies.

## Inverse model

From the GRIP checkout, install the additional Python packages:

```bash
python -m pip install '.[inverse]'
```

Obtain the [structural model source](https://github.com/zhanghaicang/carbonmatrix_public)
and the ESM2 t33 650M weights and contact-regression companion listed in its
installation instructions. Make that checkout importable with
`PYTHONPATH=/path/to/carbonmatrix_public`.

Initialize the structural model using its model configuration and local ESM resources:

```python
from argparse import Namespace
import json
from pathlib import Path

import ml_collections
import torch
from carbondesign.model.carbondesign import CarbonDesign

upstream = Path("/path/to/carbonmatrix_public")
config = json.loads(
    (upstream / "config/config_model_mrf_pair_enable_esm_sc.json").read_text()
)
config["model"]["embeddings_and_seqformer"]["lm"]["model_path"] = str(
    upstream / "params/esm2_t33_650M_UR50D.pt"
)
# The ESM resource includes an argparse.Namespace configuration.
with torch.serialization.safe_globals([Namespace]):
    backbone = CarbonDesign(ml_collections.ConfigDict(config["model"]))
```

Continue with [inverse-model loading](inverse.md#usage) using a local GRIP checkpoint.

## Attribution

The inverse model uses CarbonDesign and ESM2. Their source code and model resources
are distributed by their respective projects and remain subject to the upstream terms.

- Ren, M., Yu, C., Bu, D. & Zhang, H. Accurate and robust protein sequence design with CarbonDesign. *Nature Machine Intelligence* **6**, 536–547 (2024). [Paper](https://doi.org/10.1038/s42256-024-00838-2) · [Code](https://github.com/zhanghaicang/carbonmatrix_public).
- [ESM source and model resources](https://github.com/facebookresearch/esm).
