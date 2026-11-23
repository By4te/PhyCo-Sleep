# PhyCo-Sleep (Demo)

We present the training and evaluation code of PhyCo-Sleep on multi-view datasets. More code will be made publicly available upon paper acceptance.

## File Structure

| File | Purpose |
| --- | --- |
| `dataset.py` | Multi-view dataset wrapper `MultiViewDataset` and loaders for each dataset |
| `PhyCo_Sleep.py` | Core model: `compute_directional_conflict`, `PhyCo`, `EvidenceCollector` |
| `loss_function.py` | Loss function `get_loss` (evidential fusion training loss) |
| `import numpy as np.py` | Main training/evaluation script (`normal`) |

## Dependencies

- Python 3.8
- numpy
- scipy
- scikit-learn
- torch
- mat73 (only needed for loading `YoutubeFace_sel` and `AWA`)
