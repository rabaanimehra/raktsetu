#!/usr/bin/env python3
"""
NailScan web interface server.

Owns the USB webcam (locked colour settings), streams a live preview to the browser,
captures + analyses nail photos with the TFLite model, keeps a scan history and can
save labelled photos as new training data.

Start it with ./start_ui.sh  (or: ./venv/bin/python server.py)  then open http://localhost:8000
Set NAILSCAN_FAKE_IMAGE=/path/photo.jpg to test without a webcam.
"""
import csv, io, json, os, re, shutil, subprocess, threading, time, uuid, zipfile
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, abort

import predict_pi as pp

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
SCANS = os.path.join(HERE, "scans")
TRAIN = os.path.join(HERE, "training_data")
SETTINGS_FILE = os.path.join(HERE, "camera_settings.json")
PORT = int(os.environ.get("NAILSCAN_PORT", "8000"))
FAKE_IMAGE = os.environ.get("NAILSCAN_FAKE_IMAGE", "")
os.makedirs(SCANS, exist_ok=True)

MODEL_FILE = os.environ.get("NAILSCAN_MODEL", "anemia_model_float32.tflite")
clf = pp.AnemiaClassifier(os.path.join(pp.MODELS, MODEL_FILE), os.path.join(pp.MODELS, "model_config.json"))
clf_lock = threading.Lock()

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def parse_controls(text):
    """v4l2-ctl --list-ctrls -> {name: {min, max, step, value}}"""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)", line)
        if not m:
            continue
        vals = dict(re.findall(r"(\w+)=(-?\d+)", m.group(3)))
        out[m.group(1)] = {k: int(v) for k, v in vals.items() if k in ("min", "max", "step", "default", "value")}
    return out


class Camera:
    def __init__(self):
        self.frame = None            # latest RGB numpy frame
        self.lock = threading.Lock()
        self.status = "starting"
        self.message = ""
        self.controls = {}
        self.applied = []
        self.fps = 0.0
        self._reapply = False
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def dev(self):
        return f"/dev/video{pp.CAMERA_INDEX}"

    def apply_settings(self):
        self._reapply = True

    def _run(self):
        if FAKE_IMAGE:
            img = np.asarray(ImageOps.exif_transpose(Image.open(FAKE_IMAGE)).convert("RGB"))
            with self.lock:
                self.frame = img
            self.status, self.message, self.fps = "ok", "Test image (no webcam)", 0
            return
        while True:
            cap = cv2.VideoCapture(pp.CAMERA_INDEX, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.status = "missing"
                self.message = f"Webcam not found at {self.dev}"
                time.sleep(3)
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, pp.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pp.HEIGHT)
            try:
                self.applied = pp.lock_webcam(self.dev)
                self.controls = parse_controls(pp.list_controls(self.dev))
            except SystemExit as e:              # v4l2-ctl missing
                self.applied, self.message = [], str(e)
            self.status, self.message = "ok", self.message or "Webcam ready"
            fails, t0, n = 0, time.time(), 0
            while True:
                if self._reapply:
                    self._reapply = False
                    self.applied = pp.lock_webcam(self.dev)
                    self.controls = parse_controls(pp.list_controls(self.dev))
                ok, bgr = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        self.status, self.message = "missing", "Webcam stopped sending images"
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                with self.lock:
                    self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                n += 1
                if time.time() - t0 > 2:
                    self.fps, t0, n = n / (time.time() - t0), time.time(), 0
            cap.release()
            time.sleep(1)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


camera = Camera()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def zoom_crop(img, zoom, cx, cy):
    """Crop the region the user saw in the zoomed preview (zoom >= 1, cx/cy = view centre 0..1)."""
    w, h = img.size
    zoom = max(1.0, float(zoom))
    cw, ch = w / zoom, h / zoom
    cx = min(max(float(cx), 0.5 / zoom), 1 - 0.5 / zoom)
    cy = min(max(float(cy), 0.5 / zoom), 1 - 0.5 / zoom)
    box = (round(cx * w - cw / 2), round(cy * h - ch / 2), round(cx * w + cw / 2), round(cy * h + ch / 2))
    return img.crop(box), box


def check(name, value, status, text):
    return {"name": name, "value": value, "status": status, "text": text}


def analyse(full, zoom=1.0, cx=0.5, cy=0.5, source="camera"):
    t_start = time.perf_counter()
    crop, box = zoom_crop(full, zoom, cx, cy)
    a = np.asarray(crop).astype(np.float32)
    h, w = a.shape[:2]
    centre = a[int(.2 * h):int(.8 * h), int(.2 * w):int(.8 * w)].reshape(-1, 3)   # nail-bed centre region

    # --- model ---
    with clf_lock:
        prob, label, ms = clf.predict(crop)
    t = clf.threshold
    margin = (prob - t) / (1 - t) if prob >= t else (t - prob) / t
    confidence = "High" if margin > 0.5 else ("Moderate" if margin > 0.2 else "Low")

    # --- photo quality ---
    full_a = np.asarray(full).astype(np.float32)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean()
    red_clip = float((np.asarray(crop)[..., 0] >= 250).mean() * 100)
    gray = cv2.cvtColor(np.asarray(crop.resize((224, round(224 * h / max(w, 1))) if w else crop)), cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    cast = float(full_a[..., 0].mean() / max(full_a[..., 2].mean(), 1))
    side = min(w, h)
    checks = [
        check("Red channel clipping", f"{red_clip:.1f}%",
              "ok" if red_clip < 1 else ("warn" if red_clip < 5 else "fail"),
              "Colour detail is intact." if red_clip < 1 else
              "Some red is maxed out, so pale and pink nails look alike. Lower the exposure."),
        check("Brightness", f"{lum:.0f} / 255",
              "ok" if 70 <= lum <= 200 else ("warn" if 45 <= lum <= 225 else "fail"),
              "Well exposed." if 70 <= lum <= 200 else
              ("Too dark. Raise the exposure." if lum < 70 else "Too bright. Lower the exposure.")),
        check("Sharpness", f"{sharp:.0f}",
              "ok" if sharp >= 60 else ("warn" if sharp >= 25 else "fail"),
              "In focus." if sharp >= 60 else "Blurry. Hold the finger still and check the focus."),
        check("Colour cast (red ÷ blue)", f"{cast:.2f}",
              "ok" if cast < 1.5 else ("warn" if cast < 2.0 else "fail"),
              "Neutral lighting." if cast < 1.5 else
              "Strong red tint from the enclosure. A white floor sheet or white-balance change will help."),
        check("Nail resolution", f"{w}×{h} px",
              "ok" if side >= 150 else ("warn" if side >= 80 else "fail"),
              "Enough detail." if side >= 150 else "Zoomed in too far. Move the camera closer instead."),
    ]
    worst = "fail" if any(c["status"] == "fail" for c in checks) else ("warn" if any(c["status"] == "warn" for c in checks) else "ok")

    # --- colour analysis (centre of the crop) ---
    mean = centre.mean(0)
    hsv = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hist = {c: np.histogram(centre[:, i], bins=32, range=(0, 256))[0].tolist() for i, c in enumerate("rgb")}
    colour = {
        "mean_rgb": [round(float(v), 1) for v in mean],
        "hex": "#%02x%02x%02x" % tuple(int(v) for v in mean),
        "redness_index": round(float((mean[0] - mean[1]) / max(mean[0] + mean[1], 1)), 3),
        "saturation": round(float(hsv[:, 1].mean() / 255 * 100), 1),
        "histogram": hist,
    }

    # --- save files ---
    sid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = os.path.join(SCANS, sid)
    os.makedirs(d)
    marked = full.copy()
    marked.thumbnail((1280, 1280))
    s = marked.width / full.width
    ImageDraw.Draw(marked).rectangle([v * s for v in box], outline=(255, 255, 255), width=max(2, marked.width // 300))
    marked.save(os.path.join(d, "frame.jpg"), quality=88)
    crop.save(os.path.join(d, "crop.png"))
    pp.letterbox(crop, clf.size).save(os.path.join(d, "input.png"))

    report = {
        "id": sid,
        "time": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "result": {"label": label, "anemic": prob >= t, "probability": round(prob, 4),
                   "threshold": round(t, 4), "confidence": confidence},
        "quality": {"overall": worst, "checks": checks},
        "colour": colour,
        "capture": {"zoom": round(float(zoom), 2), "box": list(box), "frame_size": list(full.size),
                    "camera": {"exposure": pp.EXPOSURE, "white_balance_k": pp.WB_TEMPERATURE, "gain": pp.GAIN}
                    if source == "camera" else None},
        "model": {"file": MODEL_FILE, "inference_ms": round(ms, 1),
                  "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                  "test_metrics": clf.cfg.get("test_metrics_keras", {})},
        "images": {k: f"/scans/{sid}/{k}.{ext}" for k, ext in (("frame", "jpg"), ("crop", "png"), ("input", "png"))},
        "training_label": None,
    }
    json.dump(report, open(os.path.join(d, "report.json"), "w"), indent=2)
    return report


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


@app.get("/scans/<sid>/<name>")
def scan_file(sid, name):
    if not re.fullmatch(r"[\w-]+", sid):
        abort(404)
    return send_from_directory(os.path.join(SCANS, sid), name)


@app.get("/stream")
def stream():
    def gen():
        while True:
            f = camera.latest()
            if f is None:
                time.sleep(0.2)
                continue
            h, w = f.shape[:2]
            if w > 960:
                f = cv2.resize(f, (960, round(960 * h / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(1 / 15 if not FAKE_IMAGE else 0.5)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


def ctrl_range(*names):
    for n in names:
        if n in camera.controls:
            return camera.controls[n]
    return None


@app.get("/api/status")
def status():
    f = camera.latest()
    return jsonify({
        "camera": {"status": camera.status, "message": camera.message, "fps": round(camera.fps, 1),
                   "frame_size": None if f is None else [f.shape[1], f.shape[0]],
                   "locked": camera.applied, "fake": bool(FAKE_IMAGE)},
        "settings": {"exposure": pp.EXPOSURE, "white_balance_k": pp.WB_TEMPERATURE, "gain": pp.GAIN},
        "ranges": {"exposure": ctrl_range("exposure_time_absolute", "exposure_absolute"),
                   "white_balance_k": ctrl_range("white_balance_temperature"),
                   "gain": ctrl_range("gain")},
        "model": {"file": MODEL_FILE, "threshold": clf.threshold, "test_metrics": clf.cfg.get("test_metrics_keras", {}),
                  "trained_on": clf.cfg.get("trained_on")},
        "training_count": training_counts(),
    })


@app.get("/api/live")
def live():
    f = camera.latest()
    if f is None:
        return jsonify({"ok": False})
    small = f[::4, ::4].astype(np.float32)
    lum = (0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]).mean()
    return jsonify({"ok": True, "brightness": round(float(lum)),
                    "red_clip": round(float((small[..., 0] >= 250).mean() * 100), 1),
                    "cast": round(float(small[..., 0].mean() / max(small[..., 2].mean(), 1)), 2)})


@app.post("/api/camera")
def set_camera():
    d = request.get_json(force=True)
    if "exposure" in d:        pp.EXPOSURE = int(d["exposure"])
    if "white_balance_k" in d: pp.WB_TEMPERATURE = int(d["white_balance_k"])
    if "gain" in d:            pp.GAIN = int(d["gain"])
    json.dump({"EXPOSURE": pp.EXPOSURE, "WB_TEMPERATURE": pp.WB_TEMPERATURE, "GAIN": pp.GAIN},
              open(SETTINGS_FILE, "w"), indent=2)
    camera.apply_settings()
    return jsonify({"ok": True})


@app.post("/api/capture")
def capture():
    d = request.get_json(force=True) if request.data else {}
    f = camera.latest()
    if f is None:
        return jsonify({"error": "No camera image yet. Check that the webcam is connected."}), 503
    return jsonify(analyse(Image.fromarray(f), d.get("zoom", 1), d.get("cx", .5), d.get("cy", .5), "camera"))


@app.post("/api/analyze")
def analyze_upload():
    file = request.files.get("photo")
    if not file:
        return jsonify({"error": "No photo was uploaded."}), 400
    try:
        img = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
    except Exception:
        return jsonify({"error": "That file isn't an image NailScan can read. Use JPG or PNG."}), 400
    return jsonify(analyse(img, 1, .5, .5, "upload"))


@app.get("/api/history")
def history():
    items = []
    for sid in sorted(os.listdir(SCANS), reverse=True)[:60]:
        p = os.path.join(SCANS, sid, "report.json")
        if os.path.exists(p):
            r = json.load(open(p))
            items.append({"id": sid, "time": r["time"], "label": r["result"]["label"],
                          "anemic": r["result"]["anemic"], "probability": r["result"]["probability"],
                          "quality": r["quality"]["overall"], "thumb": r["images"]["crop"],
                          "training_label": r.get("training_label")})
    return jsonify(items)


@app.get("/api/scan/<sid>")
def get_scan(sid):
    p = os.path.join(SCANS, sid, "report.json")
    if not re.fullmatch(r"[\w-]+", sid) or not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="application/json")


@app.delete("/api/scan/<sid>")
def delete_scan(sid):
    p = os.path.join(SCANS, sid)
    if not re.fullmatch(r"[\w-]+", sid) or not os.path.isdir(p):
        abort(404)
    shutil.rmtree(p)
    return jsonify({"ok": True})


# --- training data -----------------------------------------------------------
WHO_HB = {"female": 12.0, "male": 13.0, "pregnant": 11.0, "child": 11.0}   # g/dL anemia cut-offs


def training_counts():
    return {c: len(os.listdir(os.path.join(TRAIN, c))) if os.path.isdir(os.path.join(TRAIN, c)) else 0
            for c in ("anemic", "non_anemic")}


@app.post("/api/label")
def label_scan():
    d = request.get_json(force=True)
    sid = d.get("id", "")
    p = os.path.join(SCANS, sid, "report.json")
    if not re.fullmatch(r"[\w-]+", sid) or not os.path.exists(p):
        return jsonify({"error": "Scan not found."}), 404
    try:
        hb = float(d["hb"])
    except (KeyError, ValueError):
        return jsonify({"error": "Enter the hemoglobin value in g/dL, for example 11.4."}), 400
    if not 3 <= hb <= 25:
        return jsonify({"error": "Hemoglobin should be between 3 and 25 g/dL. Check the value and units."}), 400
    group = d.get("group", "female")
    cut = WHO_HB.get(group, 12.0)
    cls = "anemic" if hb < cut else "non_anemic"
    os.makedirs(os.path.join(TRAIN, cls), exist_ok=True)
    fn = f"{sid}.png"
    for c in ("anemic", "non_anemic"):                       # relabelling replaces the old copy
        old = os.path.join(TRAIN, c, fn)
        if os.path.exists(old):
            os.remove(old)
    shutil.copy(os.path.join(SCANS, sid, "crop.png"), os.path.join(TRAIN, cls, fn))

    meta_path = os.path.join(TRAIN, "metadata.csv")
    rows = []
    if os.path.exists(meta_path):
        rows = [r for r in csv.DictReader(open(meta_path)) if os.path.basename(r["filename"]) != fn]
    rows.append({"filename": f"{cls}/{fn}", "person_id": d.get("person_id", "").strip() or sid,
                 "group": d.get("skin_tone", ""), "hb_g_dl": hb, "sex_group": group, "scan_id": sid})
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "person_id", "group", "hb_g_dl", "sex_group", "scan_id"])
        w.writeheader(); w.writerows(rows)

    r = json.load(open(p))
    r["training_label"] = {"class": cls, "hb": hb, "group": group, "cutoff": cut,
                           "skin_tone": d.get("skin_tone", ""), "person_id": d.get("person_id", "")}
    json.dump(r, open(p, "w"), indent=2)
    return jsonify({"ok": True, "class": cls, "cutoff": cut, "counts": training_counts()})


@app.get("/api/training.zip")
def training_zip():
    if not os.path.isdir(TRAIN):
        return jsonify({"error": "No training photos saved yet."}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(TRAIN):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.join("nailscan_training_data", os.path.relpath(full, TRAIN)))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"nailscan_training_data_{datetime.now():%Y%m%d}.zip")


if __name__ == "__main__":
    print(f"NailScan running: open http://localhost:{PORT}  (or http://<pi-ip>:{PORT} from a phone on the same Wi-Fi)")
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
