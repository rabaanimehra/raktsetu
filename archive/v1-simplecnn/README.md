# RaktSetu Version 1: SimpleCNN haemoglobin regression

The first RaktSetu model, kept here with its original commit history (1 Jul 2026) to document how the project developed.

| | |
|---|---|
| Task | Predict haemoglobin (g/dL) from a fingernail crop |
| Data | Yakimov et al. 2024, 250 adults; metadata in `archive/metadata.csv` (Hb in g/L, nail and skin bounding boxes) |
| Model | Small CNN in PyTorch (`anemia_simplecnn_edge.pth`), exported to ONNX for edge devices (`anemia_simplecnn_edge.onnx` + `.onnx.data`) |
| Result | Best validation RMSE **2.19 g/dL** on an 83-patient held-out set (from the stored training logs in the notebook) |
| Notebooks | `haemoglobin-level-from-fingernail-images.ipynb` (training), `model.ipynb` |

**Why it was replaced:** a single adult dataset with no per-patient skin-tone field could not support cross-population testing. Version 2 (`training/`, `device/models/`) switched to a classifier trained on two populations with leakage-safe splits and runs as TFLite on the Raspberry Pi.
