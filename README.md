<img src="docs/assets/grip-logo.png" alt="GRIP" width="180">

# Energy-guided de novo design of functional immunogenic neoantigens

GRIP is an energy-guided framework for designing immunopeptides for specified MHC and T-cell receptor contexts. It learns single-residue preferences and pairwise residue couplings to connect binding prediction with peptide sequence design. The study combines computational benchmarks with experimental validation of MHC-I and MHC-II binding. In the KRAS G12D system, designed peptides elicited T-cell-mediated cytotoxicity comparable to or greater than that of the native peptide in the tested TCR clonotypes.

This repository provides the sequence energy model, GRIP-CLA classifier and inverse model for structure-conditioned peptide design.

[![GRIP framework, Figure 1](docs/assets/figure-1.png)](docs/assets/figure-1.pdf)

[Website](https://zhanggroup.org/GRIP/) · [Usage](docs/usage.md) · [Model interfaces](docs/models.md) · [Weights](https://github.com/cosmoss8274/GRIP/releases/tag/v0.1.0)

## Installation

Requires Python 3.9+ and PyTorch 2.6+.

```bash
git clone https://github.com/cosmoss8274/GRIP.git
cd GRIP
python -m pip install .
```

## Load model

```python
import json
from grip_models import load_pretrained, score_record

model, config = load_pretrained("mhc-i-energy")

with open("record.json") as stream:
    record = json.load(stream)
energies = score_record(model, record, config)
print(energies)  # single, pair, total
```

Supply a peptide and its MHC pseudosequence in [the input format](docs/usage.md#prepare-a-record). The model returns single-site, pairwise and total energies. Weights are downloaded on first use and cached locally.

See [Models](docs/models.md) for classification and inverse-model usage. The [weight download](https://github.com/cosmoss8274/GRIP/releases/tag/v0.1.0) contains the MHC-I energy checkpoint; classification and inverse inference use local task-specific checkpoints.

To try input encoding without weights, run `python examples/core.py encode`. For command-line scoring, local checkpoints and offline use, see [Usage](docs/usage.md).

## License

[MIT](LICENSE). External dependencies retain their respective terms; see [Dependencies](docs/dependencies.md).
