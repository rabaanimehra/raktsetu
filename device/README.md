# RaktSetu device software (NailScan)

Everything that runs on the Raspberry Pi 5: the touchscreen interface, the webcam
capture and the TensorFlow Lite model.

## Install

```bash
git clone https://github.com/rabaanimehra/raktsetu.git ~/RaktSetu
bash ~/RaktSetu/device/install.sh
```

Add `--autostart` to the second command to open NailScan full screen every time the Pi boots.
The installer creates `device/venv`, installs all libraries, tests the model and adds a
NailScan icon to the desktop and app menu. Reboot once afterwards so the webcam permission
takes effect.

## Update

```bash
cd ~/RaktSetu && git pull && bash device/install.sh
```

Your scans, training photos and camera settings are not tracked by git, so updating never touches them.

## Use

- Double-click **NailScan** on the desktop, or run `~/RaktSetu/device/start_ui.sh` (`--kiosk` for full screen, Alt+F4 to quit).
- From a phone on the same Wi-Fi: `http://<pi-ip-address>:8000`.

1. Put a finger in the enclosure. Zoom and drag until **one nail fills the frame**. The zoomed area is exactly what gets analysed.
2. Check **Live photo check**: red clipping should be under 1%. If not, lower the exposure in **Camera settings** (saved automatically).
3. Tap the round button to get the report: result on a pink-to-pale scale, photo-quality checks, nail colour analysis, scan details. **Save PDF** saves it.
4. If the person has a recent blood test, enter their hemoglobin under **Add to training data**. The labelled photos go to `device/training_data/` and can be downloaded as a zip for retraining in `training/`.

## Command line

| Command | What it does |
|---|---|
| `./check_camera.sh` | One webcam photo + colour / red-clipping report |
| `./run.sh --webcam` | One photo + prediction in the terminal |
| `./run.sh --image photo.jpg` | Predict on a saved photo |
| `./run.sh --list-controls` | Webcam settings it supports |

## Files

| Path | Purpose |
|---|---|
| `server.py`, `static/` | Web interface (Flask server + single-page UI) |
| `predict_pi.py` | Model and webcam code, also the CLI. Webcam defaults at the top |
| `models/` | `anemia_model_float32.tflite` (default), `anemia_model_int8.tflite`, `model_config.json` |
| `install.sh`, `start_ui.sh`, `run.sh`, `check_camera.sh` | Install and launch scripts |
| `scans/`, `training_data/`, `camera_settings.json`, `server.log` | Created on the Pi, ignored by git |

Use the smaller model with `NAILSCAN_MODEL=anemia_model_int8.tflite ./start_ui.sh`.

## Troubleshooting

- **Webcam not found**: check the USB cable. If the Pi has other cameras, set `CAMERA_INDEX = 1` at the top of `predict_pi.py`.
- **Webcam permission error**: reboot once after installing.
- **Interface won't load**: read `device/server.log`.

> Screening prototype only, not a medical diagnosis. Confirm with a hemoglobin blood test.
