<div align="center">

# RaktSetu

### Non-invasive anaemia screening from a fingernail photo

**A portable, offline, Raspberry Pi device that photographs the nail bed and flags likely anaemia, with no needle, no reagents and no lab.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow&logoColor=white)](https://ai.google.dev/edge/litert)
[![Raspberry Pi 5](https://img.shields.io/badge/Raspberry%20Pi-5-A22846?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![Status](https://img.shields.io/badge/status-working%20prototype-B3202F)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

<img src="dashboard/proto-front.jpg" width="31%" alt="RaktSetu prototype, front"> <img src="dashboard/proto-angle-open.jpg" width="31%" alt="RaktSetu prototype, open chamber"> <img src="dashboard/proto-top.jpg" width="31%" alt="RaktSetu prototype, top">

**[Evidence and proof of work →](PROOF.md)** &nbsp;·&nbsp; **[Run it on a Pi →](docs/RaktSetu_Pi_Setup.txt)** &nbsp;·&nbsp; **[Project site →](dashboard/index.html)**

</div>

---

## At a glance

| | |
|---|---|
| **Problem** | 58.7% of women and 71.1% of young children in Punjab are anaemic (NFHS-5). Confirming it needs a blood test, so most people are never checked. |
| **What RaktSetu does** | Finger goes in a light-sealed chamber, a fixed USB camera photographs the nail bed, an on-device model returns an anaemia flag in seconds. |
| **Model** | EfficientNet-B0, fine-tuned, exported to TensorFlow Lite (float32 and int8) |
| **Test result** | AUC **0.937**, accuracy **83.7%**, sensitivity **79.8%**, specificity **95.1%** on **485** unseen images |
| **Runs on** | Raspberry Pi 5 + touchscreen + USB webcam, fully offline. Images never leave the device. |
| **Stage** | Working prototype; next step is on-device validation against lab haemoglobin |

> RaktSetu is a screening aid, not a diagnostic device. A "possible anaemia" result must be confirmed with a haemoglobin blood test.

---

## How it works

```
 finger in light-sealed chamber, USB webcam with locked colour settings
            │
            ▼
 zoom / crop to one nail ........ operator frames the nail on the touchscreen
            │
            ▼
 photo-quality checks ........... red clipping, brightness, sharpness, colour cast
            │
            ▼
 EfficientNet-B0 (TFLite) ....... 224 × 224 nail crop → anaemia probability
            │
            ▼
 on-device report ............... result, confidence, colour analysis, PDF export
```

If the person has a recent blood test, the operator can enter their haemoglobin and the photo is saved as labelled training data for retraining on device-captured images.

---

## Repository map

```
raktsetu/
├── PROOF.md                      evidence index: build timeline, results, hardware, what's next
├── device/                       software that runs on the Raspberry Pi 5 (NailScan)
│   ├── install.sh                one-command installer
│   ├── server.py                 Flask server: webcam, analysis, history, reports
│   ├── predict_pi.py             model + webcam code, also a CLI
│   ├── static/                   touchscreen interface
│   └── models/                   anaemia_model_float32/int8.tflite + model_config.json (metrics)
├── training/
│   └── anemia_nailbed_training.ipynb    Colab notebook that trains and exports the model
├── dashboard/                    project website + prototype photos
├── docs/
│   └── RaktSetu_Pi_Setup.txt     step-by-step Pi setup guide
└── archive/
    └── v1-simplecnn/             first model (SimpleCNN Hb regression), kept with its history
```

---

## Results

From `device/models/model_config.json`, model trained 23 Oct 2026, decision threshold tuned on the validation set for ≥85% sensitivity.

| Model | n (test) | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| EfficientNet-B0, Keras | 485 | 0.937 | 83.7% | 79.8% | 95.1% |
| TFLite float32 (on Pi, default) | 485 | 0.937 | 83.7% | 79.8% | 95.1% |
| TFLite int8 (on Pi, small) | 485 | 0.920 | 83.9% | 83.1% | 86.2% |

Split: 2,417 train / 483 validation / 485 test. Augmented copies and photos of the same person are always kept in the same split, so test images are truly unseen.

These numbers are on public-dataset images, not photos taken in the RaktSetu chamber. On-device validation is the next milestone (see [PROOF.md](PROOF.md#whats-next)).

---

## Datasets

1. **Asare et al., 2022, Mendeley Data.** Detection of Anemia using Colour of the Fingernails. 710 images of children ≤5 years from 10 hospitals in Ghana, labelled against lab haemoglobin. [doi.org/10.17632/2xx4j3kjg2.1](https://doi.org/10.17632/2xx4j3kjg2.1)
2. **Yakimov et al., 2024, Scientific Data.** Human skin and fingernail images for non-invasive Hb assessment. 250 adults aged 18 to 95, exact Hb and Fitzpatrick skin type per patient. [doi.org/10.1038/s41597-024-03895-9](https://doi.org/10.1038/s41597-024-03895-9)

Combining a paediatric, classification-labelled set with an adult, continuous-Hb set exposes the model to two common failure modes of colour-based screening: age and skin tone.

---

## Getting started

**On a Raspberry Pi 5** (full guide in [`docs/RaktSetu_Pi_Setup.txt`](docs/RaktSetu_Pi_Setup.txt)):

```bash
git clone https://github.com/rabaanimehra/raktsetu.git ~/RaktSetu
bash ~/RaktSetu/device/install.sh
```

Reboot once, then open **NailScan** from the desktop. Details in [`device/README.md`](device/README.md).

**Retrain the model:** open `training/anemia_nailbed_training.ipynb` in Google Colab with a T4 GPU and choose Run all. See [`training/README.md`](training/README.md).

---

## Team and credits

- **Rabaani Mehra**, project lead: problem framing, enclosure and device design, community screening, validation plan
- Code contributions from a collaborator who has asked not to be named
- Datasets by Asare et al. (2022) and Yakimov et al. (2024)

Full commit history is preserved in this repository.

---

## Citing the datasets

```
Asare, J. et al. (2022). Detection of Anemia using Colour of the Fingernails
Image Dataset. Mendeley Data, V1. https://doi.org/10.17632/2xx4j3kjg2.1

Yakimov, B. et al. (2024). Dataset of human skin and fingernails images for
non-invasive haemoglobin level assessment. Scientific Data, 11, 1278.
https://doi.org/10.1038/s41597-024-03895-9
```
