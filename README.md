# PhyCo: Multi-View Evidential Fusion Classification

A multi-view classification framework based on evidence theory (Dirichlet distribution). Each view is processed by an evidence collector that outputs a non-negative evidence vector. View-wise "directional conflict" is then computed to derive conflict-aware weights, and the weighted evidence is fused into a final evidence vector for classification.

## File Structure

| File | Purpose |
| --- | --- |
| `dataset.py` | Multi-view dataset wrapper `MultiViewDataset` and loaders for each dataset |
| `PhyCo_Sleep.py` | Core model: `compute_directional_conflict`, `PhyCo`, `EvidenceCollector` |
| `loss_function.py` | Loss function `get_loss` (evidential fusion training loss) |
| `import numpy as np.py` | Main training/evaluation script (`normal`) |

## Dependencies

- Python 3.x
- numpy
- scipy
- scikit-learn
- torch
- mat73 (only needed for loading `YoutubeFace_sel` and `AWA`)

```bash
pip install numpy scipy scikit-learn torch mat73
