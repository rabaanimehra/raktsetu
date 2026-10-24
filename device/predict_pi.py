#!/usr/bin/env python3
"""
RaktSetu - nail-bed anemia screening on Raspberry Pi 5 (USB webcam). Model + webcam code and CLI.

SETUP (once, on the Pi):
    bash install.sh        # creates ./venv and installs everything

FILES needed in the models/ folder next to this script:
    anemia_model_float32.tflite   (or anemia_model_int8.tflite)
    model_config.json

USAGE:
    ./run.sh --list-controls            # see which settings your webcam supports
    ./run.sh --webcam --check           # take a photo, report red clipping, save it (no prediction)
    ./run.sh --webcam                   # take a photo with locked settings and classify it
    ./run.sh --image nail.jpg           # classify a saved photo
    ./run.sh --webcam --ref 20 20 80 80 # also colour-correct using a white sticker at that box

This is a SCREENING prototype, not a medical diagnosis.
"""
import argparse, json, os, re, subprocess, time
import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# ---------------------------------------------------------------------------
# WEBCAM SETTINGS  - tune these once for your enclosure, then never change them
# ---------------------------------------------------------------------------
CAMERA_INDEX  = 0          # /dev/video0
WIDTH, HEIGHT = 1280, 720
WB_TEMPERATURE = 5000      # Kelvin. Lower (e.g. 4000) = less red/warmer correction, higher = more blue
EXPOSURE       = 150       # lower this if --check reports red clipping (units depend on the camera)
GAIN           = 0         # keep low; raises noise
# Optional nail crop box (left, top, right, bottom) in the full webcam frame. None = use whole frame.
NAIL_BOX = None

# Settings saved from the NailScan web interface override the defaults above
_SETTINGS_FILE = os.path.join(HERE, "camera_settings.json")
if os.path.exists(_SETTINGS_FILE):
    try:
        _saved = json.load(open(_SETTINGS_FILE))
        globals().update({k: v for k, v in _saved.items()
                          if k in ("CAMERA_INDEX", "WIDTH", "HEIGHT", "WB_TEMPERATURE", "EXPOSURE", "GAIN")})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Webcam control (v4l2-ctl).  Names differ between kernels, so we pick what exists.
# ---------------------------------------------------------------------------
def list_controls(dev):
    try:
        return subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls"],
                              capture_output=True, text=True).stdout
    except FileNotFoundError:
        raise SystemExit("v4l2-ctl not found. Run: sudo apt install -y v4l-utils")


def lock_webcam(dev):
    """Turn off auto white balance / auto exposure so every photo has identical colour."""
    ctrls = list_controls(dev)
    has = lambda n: re.search(rf"^\s*{n}\b", ctrls, re.M) is not None
    want = []
    # white balance
    if has("white_balance_automatic"):        want.append("white_balance_automatic=0")
    elif has("white_balance_temperature_auto"): want.append("white_balance_temperature_auto=0")
    if has("white_balance_temperature"):      want.append(f"white_balance_temperature={WB_TEMPERATURE}")
    # exposure (1 = manual mode on UVC webcams)
    if has("auto_exposure"):                  want.append("auto_exposure=1")
    elif has("exposure_auto"):                want.append("exposure_auto=1")
    if has("exposure_dynamic_framerate"):     want.append("exposure_dynamic_framerate=0")
    elif has("exposure_auto_priority"):       want.append("exposure_auto_priority=0")
    if has("exposure_time_absolute"):         want.append(f"exposure_time_absolute={EXPOSURE}")
    elif has("exposure_absolute"):            want.append(f"exposure_absolute={EXPOSURE}")
    if has("gain"):                           want.append(f"gain={GAIN}")
    # order matters: switch auto modes off before setting manual values
    for c in want:
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", c], capture_output=True)
    if not want:
        print("! Could not find white-balance/exposure controls on this webcam - colours may drift.")
    return want


def capture_from_webcam():
    import cv2
    dev = f"/dev/video{CAMERA_INDEX}"
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {dev}. Check the cable / try CAMERA_INDEX = 1.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    applied = lock_webcam(dev)                 # apply AFTER opening (opening can reset settings)
    for _ in range(20):                        # let the new settings take effect
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Webcam returned no image.")
    print("Webcam locked with:", ", ".join(applied) if applied else "(nothing)")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Image processing (letterbox MUST match the training notebook)
# ---------------------------------------------------------------------------
def letterbox(img, size=224):
    """Resize keeping the aspect ratio, then pad with black to size x size."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def white_balance(img, ref_box):
    """Scale colours so the white reference sticker inside ref_box becomes neutral."""
    a = np.asarray(img, dtype=np.float32)
    l, t, r, b = ref_box
    ref = np.median(a[t:b, l:r].reshape(-1, 3), axis=0)
    a = a * (ref.mean() / np.maximum(ref, 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def colour_report(img, box=None):
    a = np.asarray(img if box is None else img.crop(box))
    names = "RGB"
    print("\nColour check" + (" (nail box)" if box else " (whole frame)") + ":")
    for c in range(3):
        ch = a[..., c]
        print(f"  {names[c]}: mean {ch.mean():6.1f} | clipped (>=250) {100*(ch >= 250).mean():5.1f}%")
    red_clip = 100 * (a[..., 0] >= 250).mean()
    if red_clip > 1:
        print(f"  ⚠️  Red is clipped in {red_clip:.1f}% of pixels -> lower EXPOSURE (or GAIN) at the top of this script.")
    elif a[..., 0].mean() > 1.6 * a[..., 2].mean():
        print("  ⚠️  Very strong red cast. A white sheet on the enclosure floor will help the most.")
    else:
        print("  ✅ No red clipping.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_interpreter(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=model_path, num_threads=4)   # Pi 5 has 4 cores
    it.allocate_tensors()
    return it


class AnemiaClassifier:
    def __init__(self, model_path, config_path):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.size = self.cfg["img_size"]
        self.threshold = self.cfg["threshold"]
        self.interp = load_interpreter(model_path)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict(self, pil_img):
        x = np.asarray(letterbox(pil_img, self.size), dtype=np.float32)[None]  # pixels 0-255
        self.interp.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        t0 = time.perf_counter()
        self.interp.invoke()
        ms = (time.perf_counter() - t0) * 1000
        prob = float(np.squeeze(self.interp.get_tensor(self.out["index"])))
        label = "ANEMIC (possible)" if prob >= self.threshold else "NOT ANEMIC"
        return prob, label, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="path to a nail-bed photo")
    ap.add_argument("--webcam", action="store_true", help="capture from the USB webcam")
    ap.add_argument("--check", action="store_true", help="only capture + colour report, no prediction")
    ap.add_argument("--list-controls", action="store_true", help="print the webcam's adjustable settings")
    ap.add_argument("--ref", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                    help="box of a white reference sticker in the frame, for colour correction")
    ap.add_argument("--model", default=os.path.join(MODELS, "anemia_model_float32.tflite"))
    ap.add_argument("--config", default=os.path.join(MODELS, "model_config.json"))
    ap.add_argument("--save", default=os.path.join(HERE, "last_capture.jpg"), help="where to save the photo")
    a = ap.parse_args()

    if a.list_controls:
        print(list_controls(f"/dev/video{CAMERA_INDEX}")); return
    if not a.image and not a.webcam:
        ap.error("give --image PATH or --webcam")

    img = capture_from_webcam() if a.webcam else ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
    img.save(a.save)
    print(f"Saved photo: {a.save}  ({img.width}x{img.height})")

    colour_report(img)
    if NAIL_BOX:
        colour_report(img, NAIL_BOX)
    if a.ref:
        img = white_balance(img, a.ref)
        img.save(os.path.splitext(a.save)[0] + "_corrected.jpg")
        print("Colour-corrected with the white reference sticker.")
    if a.check:
        return

    if NAIL_BOX:
        img = img.crop(NAIL_BOX)
    clf = AnemiaClassifier(a.model, a.config)
    prob, label, ms = clf.predict(img)
    print(f"\nResult      : {label}")
    print(f"Probability : {prob:.3f}  (threshold {clf.threshold:.3f})")
    print(f"Inference   : {ms:.1f} ms")
    print("Screening only - confirm with a hemoglobin blood test.")


if __name__ == "__main__":
    main()
