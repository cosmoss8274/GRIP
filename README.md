<img src="docs/assets/grip-logo.png" alt="GRIP" width="180">

**Energy-guided de novo design of functional immunogenic neoantigens**

GRIP learns how peptide sequences interact with MHCs and T-cell receptors to guide immunopeptide design. This repository contains the core model and sequence-scoring code.

[![GRIP core model, Figure 1A](docs/assets/figure-1a.png)](docs/assets/figure-1.pdf)

## Installation

Requires Python 3.9+ and PyTorch 2.6+.

```bash
git clone https://github.com/cosmoss8274/GRIP.git
cd GRIP
python -m pip install .
```

## Usage

Trained weights are not included in this repository. See the [usage guide](docs/usage.md) for loading a model and scoring peptides with their MHC or TCR context.

The model returns single-residue, pairwise and total energies.

## License

[MIT](LICENSE).
