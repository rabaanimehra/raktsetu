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
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow&logoColor=white)](https://ai.google.dev/edge/litert)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-A22846?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![Status](https://img.shields.io/badge/status-research%20prototype-B3202F)]()
[![License](https://img.shields.io/badge/license-unspecified-lightgrey)]()

</div>

---

## 📖 Overview

**RaktSetu** (Hindi, *"blood bridge"*) is a research project exploring whether anemia can be screened **non-invasively**, using just a photograph of a fingernail bed.

Pallor of the nail bed is a long-recognised clinical sign of anemia. RaktSetu trains deep learning models to read that sign automatically — estimating haemoglobin level and anemia likelihood from a standardised nail bed image, with no blood draw, no reagents, and no lab required.

This repository holds the whole project: the model training notebook, the trained model, the software that runs on the Raspberry Pi device, and the project website.

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
   finger in the enclosure, USB webcam (locked colour settings)
            │
            ▼
   ┌──────────────────────────┐
   │ zoom / crop to one nail   │   operator frames the nail on the touchscreen
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ photo-quality checks      │   red clipping, brightness, sharpness, colour cast
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ EfficientNet-B0 (TFLite)  │   224 × 224 nail crop → anemia probability
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ report on the device      │   result, confidence, colour analysis, PDF
   └──────────────────────────┘
```

Everything runs offline on the Raspberry Pi 5. Images never leave the device.

---

## 📂 Repository structure

```
Rakt-Setu-/
├── device/                      # runs on the Raspberry Pi 5
│   ├── install.sh               # one-command installer
│   ├── start_ui.sh              # opens the touchscreen interface
│   ├── server.py                # Flask server: webcam, analysis, history
│   ├── predict_pi.py            # model + webcam code, command-line tool
│   ├── static/                  # interface (index.html, fonts, icon)
│   └── models/                  # anemia_model_float32/int8.tflite, model_config.json
├── training/
│   └── anemia_nailbed_training.ipynb   # Colab notebook that trains the model
├── dashboard/                   # project website and prototype photos
├── LICENSE
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
- Used for training alongside the Ghana set, so the model sees adult and lighter-skinned nails; each nail is cropped from the hand photo and labelled anemic when Hb is below 120 g/L
- 🔗 [doi.org/10.1038/s41597-024-03895-9](https://doi.org/10.1038/s41597-024-03895-9)

Using a **paediatric, classification-labelled** dataset alongside an **adult, continuous-Hb** dataset lets the model be checked for two different failure modes at once: poor generalisation across age groups, and poor generalisation across skin tone — both common weak points in colour-based health-screening models.

---

## 🏗️ Model

`training/anemia_nailbed_training.ipynb` fine-tunes an ImageNet-pretrained **EfficientNet-B0** to classify a nail-bed crop as anemic or non-anemic, then exports it to **TensorFlow Lite** for the Raspberry Pi.

- **Input:** 224 × 224 RGB nail crop, aspect ratio kept (letterbox)
- **Training:** frozen-backbone phase, then fine-tuning of the top layers; class weights; augmentation that never alters hue or saturation, since colour is the signal
- **Honest evaluation:** near-duplicate (augmented) images and photos of the same person are kept in the same split, so test images are truly unseen
- **Decision threshold:** chosen on the validation set to catch at least 85% of anemic cases

**Test results** on 485 unseen public-dataset images:

| AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| 0.937 | 83.7% | 79.8% | 95.1% |

These figures come from the public datasets, not from photos taken in the RaktSetu enclosure. Validation on the device is the next step, and the interface can save labelled photos for exactly that.

---

## 🚀 Getting started

### Run it on the Raspberry Pi 5

```bash
git clone https://github.com/rabaanimehra/raktsetu.git ~/RaktSetu
bash ~/RaktSetu/device/install.sh
```

Then reboot once and double-click **NailScan** on the desktop. Full details are in [`device/README.md`](device/README.md).

To update later: `cd ~/RaktSetu && git pull && bash device/install.sh`

### Retrain the model

Open `training/anemia_nailbed_training.ipynb` in Google Colab with a T4 GPU and choose **Run all**. See [`training/README.md`](training/README.md).

---

## 🛣️ Roadmap

- [x] Train on both the Ghana and Yakimov datasets with leakage-safe splits
- [x] TFLite model running offline on the Raspberry Pi 5
- [x] Touchscreen interface with live preview, quality checks and reports
- [ ] Collect labelled photos with the device and retrain on them
- [ ] Validate against blood-test hemoglobin across skin tones and age groups
- [ ] Reduce the red tint from the enclosure (white floor insert, colour calibration)

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
