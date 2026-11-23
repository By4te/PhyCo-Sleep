# PhyCo-Sleep (Demo)

We present the training and evaluation code of PhyCo-Sleep on multi-view datasets. More code will be made publicly available upon paper acceptance.

## File Structure

| File | Purpose |
| --- | --- |
| `datasets.py` | Multi-view dataset wrapper `MultiViewDataset` and loaders for each dataset |
| `model.py` | Core model: `compute_directional_conflict`, `PhyCo`, `EvidenceCollector` |
| `loss function.py` | Loss function `get_loss` (evidential fusion training loss) |
| `main.py` | Main training/evaluation script (`normal`) |

## 
## Dependencies

- Python 3.8
- numpy
- scipy
- scikit-learn
- torch
- mat73 (only needed for loading `YoutubeFace_sel` and `AWA`)

## Acknowledgements
The experiments in this work are conducted on publicly available multi-view datasets, including Caltech, Leaves, LandUse, Scene, CCV, Fashion, NUS-OBJ, and AWA. These datasets cover a variety of domains (object images, plant leaf images, land-use data, scene images, web images, fashion images, object images, and animal images) and provide multiple heterogeneous feature views per sample, which are well suited for evaluating multi-view classification methods.

We thank the authors and maintainers of these datasets for making their data publicly available. The datasets used in this work can be accessed at [multi-view datasets](https://drive.google.com/drive/folders/1U3aVU7YZDi-l7O34tos_mQig16JjnYmL).
