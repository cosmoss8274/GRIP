<img src="docs/assets/grip-logo.png" alt="GRIP" width="180">

# Energy-guided de novo design of functional immunogenic neoantigens

GRIP is an energy-guided framework for designing immunopeptides for specified MHC and T-cell receptor contexts. It learns single-residue preferences and pairwise residue couplings to connect binding prediction with peptide sequence design. The study combines computational benchmarks with experimental validation of MHC-I and MHC-II binding. In the KRAS G12D system, designed peptides elicited T-cell-mediated cytotoxicity comparable to or greater than that of the native peptide in the tested TCR clonotypes.

This repository provides the core neural model, pretrained MHC-I energy weights and a Python interface for sequence scoring.

[![GRIP framework, Figure 1](docs/assets/figure-1.png)](docs/assets/figure-1.pdf)

[Website](https://zhanggroup.org/GRIP/) · [Usage](docs/usage.md) · [Models](https://github.com/cosmoss8274/GRIP/releases/tag/v0.1.0)

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

The first call downloads the model and its configuration; later calls use the local cache. Supply a peptide and the 34-residue MHC pseudosequence in [the original input format](docs/usage.md#prepare-a-record). The returned values are model energies. This checkpoint is the MHC-I base energy model; task-specific classifiers and peptide sampling are outside this package.

To try input encoding without weights, run `python examples/core.py encode`. For command-line scoring, local checkpoints and offline use, see [Usage](docs/usage.md).

## License

[MIT](LICENSE).
