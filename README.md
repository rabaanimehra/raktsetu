```
██████╗  █████╗ ██╗  ██╗████████╗███████╗███████╗████████╗██╗   ██╗
██╔══██╗██╔══██╗██║ ██╔╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║   ██║
██████╔╝███████║█████╔╝    ██║   ███████╗█████╗     ██║   ██║   ██║
██╔══██╗██╔══██║██╔═██╗    ██║   ╚════██║██╔══╝     ██║   ██║   ██║
██║  ██║██║  ██║██║  ██╗   ██║   ███████║███████╗   ██║   ╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝   ╚═╝    ╚═════╝
```

<div align="center">

### A bridge between a fingernail and a healthier life 🩸

**Portable, AI-based nail bed imaging for non-invasive anemia screening.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DL%20Framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Edge%20Export-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Status](https://img.shields.io/badge/status-research%20prototype-B3202F)]()
[![License](https://img.shields.io/badge/license-unspecified-lightgrey)]()

</div>

---

## 📖 Overview

**RaktSetu** (Hindi, *"blood bridge"*) is a research project exploring whether anemia can be screened **non-invasively**, using just a photograph of a fingernail bed.

Pallor of the nail bed is a long-recognised clinical sign of anemia. RaktSetu trains deep learning models to read that sign automatically — estimating haemoglobin level and anemia likelihood from a standardised nail bed image, with no blood draw, no reagents, and no lab required.

This repository holds the **machine learning core** of the project: data pipelines, model notebooks, and exported model artifacts for edge deployment.

> ⚠️ **Disclaimer:** RaktSetu is a research prototype and a *screening aid*, not a diagnostic device. It is designed to help flag people who should get a confirmatory blood test — it does not replace clinical evaluation.

---

## 🩺 Why this matters

| | |
|---|---|
| **57%** | of women aged 15–49 in India are anemic |
| **67.1%** | of children aged 6–59 months in India are anemic |
| **52.2%** | of pregnant women aged 15–49 in India are anemic |

*Source: National Family Health Survey-5 (NFHS-5), 2019–21, Ministry of Health and Family Welfare, Government of India.*

Globally, the WHO estimates anemia affects **~40% of children** aged 6–59 months and **~30% of women** aged 15–49. Standard diagnosis — a venous or finger-prick blood test — is accurate but requires needles, consumables, and lab access that is often out of reach in low-resource settings. A fast, non-invasive first check can help close that gap.

---

## 🧠 How it works (pipeline)

```
   finger / nail bed photo
            │
            ▼
   ┌─────────────────────┐
   │  crop to nail region │   (bounding-box extraction)
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐
   │   SimpleCNN model    │   conv → pool → conv → pool → FC → FC
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐
   │ Hb estimate / result │   exported to ONNX for edge inference
   └─────────────────────┘
```

---

## 📂 Repository structure

```
Anemia-prediction/
├── archive/                                     # dataset (metadata + cropped images)
├── model.ipynb                                  # model training notebook (SimpleCNN, PyTorch)
├── haemoglobin-level-from-fingernail-images.ipynb  # data exploration & pipeline notebook
├── anemia_simplecnn_edge.pth                    # trained PyTorch checkpoint
├── anemia_simplecnn_edge.onnx                   # exported ONNX model (edge-ready)
├── anemia_simplecnn_edge.onnx.data              # external ONNX weights
└── README.md
```

---

## 🗂️ Datasets

RaktSetu draws on two complementary, independently collected datasets.

### 1. Primary — Ghana fingernail anemia dataset
**Detection of Anemia using Colour of the Fingernails Image Dataset**, Asare et al., 2022, *Mendeley Data*

- 710 original images — **426 anemic**, **284 non-anemic**
- 4,260 images after augmentation
- Collected from children **aged 5 and under**, across **10 hospitals in Ghana**
- Each image labelled anemic / non-anemic against a **laboratory haemoglobin value**
- 🔗 [data.mendeley.com/datasets/2xx4j3kjg2/1](https://data.mendeley.com/datasets/2xx4j3kjg2/1)

### 2. Secondary (validation) — Skin & fingernail Hb dataset
**Dataset of Human Skin and Fingernails Images for Non-Invasive Haemoglobin Level Assessment**, Yakimov et al., 2024, *Scientific Data*

- 250 adult patients — **128 male**, **122 female**, aged **18–95**
- Each patient has an **exact haemoglobin value** and a **Fitzpatrick skin type** (Type 1 – very light, to Type 6 – very dark)
- Used to validate generalisation across skin tones and adult physiology
- 🔗 [doi.org/10.1038/s41597-024-03895-9](https://doi.org/10.1038/s41597-024-03895-9)

Using a **paediatric, classification-labelled** dataset alongside an **adult, continuous-Hb** dataset lets the model be checked for two different failure modes at once: poor generalisation across age groups, and poor generalisation across skin tone — both common weak points in colour-based health-screening models.

---

## 🏗️ Model

The current baseline (`model.ipynb`) is a compact CNN built for edge deployment:

```
Input:  3 × 64 × 64 RGB nail-bed crop
  │
  ├─ Conv(3→32, 3×3) → ReLU → MaxPool(2×2)
  ├─ Conv(32→64, 3×3) → ReLU → MaxPool(2×2)
  ├─ Flatten (64×16×16)
  ├─ FC(16384 → 128) → ReLU
  └─ FC(128 → 1)              # haemoglobin regression output
```

- **Framework:** PyTorch, exported to **ONNX** for lightweight/edge inference
- **Training:** Adam optimiser, MSE loss, batch size 32
- **Status:** early experimental baseline, trained on the validation dataset — accuracy and generalisation are still being actively improved, and the classification pipeline on the primary Ghana dataset is in progress

This is a **work in progress**, not a finished, validated model. Numbers here are from early experiments and should not be treated as a performance claim.

---

## 🚀 Getting started

```bash
git clone https://github.com/TanmoyD24/Anemia-prediction.git
cd Anemia-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision onnx pandas numpy scikit-learn scikit-image pillow jupyter

jupyter notebook model.ipynb
```

To run inference with the exported edge model:

```python
import onnxruntime as ort

session = ort.InferenceSession("anemia_simplecnn_edge.onnx")
# feed a preprocessed 1×3×64×64 nail-bed image tensor
output = session.run(None, {"input": image_tensor})
```

---

## 🛣️ Roadmap

- [ ] Finish wiring the primary Ghana dataset into the training pipeline
- [ ] Move from Hb regression to a combined regression + anemic/non-anemic classification head
- [ ] Cross-validate across both datasets (paediatric ↔ adult, skin tone range)
- [ ] Quantize / optimise the ONNX model further for low-power edge hardware
- [ ] Package inference into a simple CLI / app for the physical RaktSetu device

---

## 📜 Citing the datasets

```
Asare, J. et al. (2022). Detection of Anemia using Colour of the Fingernails
Image Dataset. Mendeley Data, V1. https://doi.org/10.17632/2xx4j3kjg2.1

Yakimov, B. et al. (2024). Dataset of human skin and fingernails images for
non-invasive haemoglobin level assessment. Scientific Data, 11, 1278.
https://doi.org/10.1038/s41597-024-03895-9
```

---

<div align="center">

Built to make a first anemia check as simple as taking a photo.

</div>
