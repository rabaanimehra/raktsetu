#!/usr/bin/env python3
"""
Nail-bed anemia screening - Raspberry Pi 5 inference script.

SETUP (once, on the Pi):
    sudo apt install -y python3-picamera2        # only if you use the Pi camera
    python3 -m venv --system-site-packages ~/anemia-env
    source ~/anemia-env/bin/activate
    pip install ai-edge-litert pillow numpy
    # if ai-edge-litert has no wheel for your Python, use: pip install tensorflow

FILES needed in the same folder as this script:
    anemia_model_float32.tflite   (or anemia_model_int8.tflite)
    model_config.json

USAGE:
    python predict_pi.py --image nail.jpg
    python predict_pi.py --camera              # capture one photo from the Pi camera
    python predict_pi.py --image nail.jpg --model anemia_model_int8.tflite

This is a SCREENING prototype, not a medical diagnosis.
"""
import argparse, json, os, time
import numpy as np
from PIL import Image, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))


def letterbox(img, size=224):
    """Resize keeping the aspect ratio, then pad with black to size x size.
    The SAME function is used on the Raspberry Pi, so training and deployment match."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


class AnemiaClassifier:
    def __init__(self, model_path, config_path):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.size = self.cfg["img_size"]
        self.threshold = self.cfg["threshold"]
        self.interp = Interpreter(model_path=model_path, num_threads=4)  # Pi 5 has 4 cores
        self.interp.allocate_tensors()
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


def capture_from_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    time.sleep(2)                         # let exposure / white balance settle
    img = cam.capture_image("main")       # PIL image with correct RGB order
    cam.stop()
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="path to a nail-bed photo")
    ap.add_argument("--camera", action="store_true", help="capture from the Pi camera")
    ap.add_argument("--model", default=os.path.join(HERE, "anemia_model_float32.tflite"))
    ap.add_argument("--config", default=os.path.join(HERE, "model_config.json"))
    ap.add_argument("--save", help="save the captured/input image here")
    a = ap.parse_args()

    if not a.image and not a.camera:
        ap.error("give --image PATH or --camera")
    img = capture_from_camera() if a.camera else Image.open(a.image)
    if a.save:
        img.save(a.save)

    clf = AnemiaClassifier(a.model, a.config)
    prob, label, ms = clf.predict(img)
    print(f"Result      : {label}")
    print(f"Probability : {prob:.3f}  (threshold {clf.threshold:.3f})")
    print(f"Inference   : {ms:.1f} ms")
    print("Screening only - confirm with a hemoglobin blood test.")


if __name__ == "__main__":
    main()
