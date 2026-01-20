from ultralytics import YOLO
from pathlib import Path
import re

# ===== CONFIG =====
MODEL_PATH = "runs/detect/train9/weights/best.pt"
SOURCE_DIR = Path("auto_label/images/unlabeled")
OUT_DIR = Path("auto_label/predictions")
CONF_THRES = 0.25

CLASS_MAP = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5
}

# ==================

model = YOLO(MODEL_PATH)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "labels").mkdir(exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)

def extract_class(filename):
    name = filename.lower()
    for cls in CLASS_MAP:
        if cls in name:
            return cls
    return None

results = model.predict(
    source=str(SOURCE_DIR),
    imgsz=640,
    conf=CONF_THRES,
    save=True,
    project=str(OUT_DIR),
    name="images",
    save_conf=True
)

for r in results:
    img_path = Path(r.path)
    fname = img_path.name

    true_class = extract_class(fname)

    if true_class is None:
        print(f"[WARN] No class found in filename: {fname}")
        continue

    class_id = CLASS_MAP[true_class]
    label_path = OUT_DIR / "labels" / (img_path.stem + ".txt")

    filtered_lines = []

    if r.boxes is not None:
        for box in r.boxes:
            conf = float(box.conf[0])
            xywh = box.xywhn[0].tolist()

            # force class to filename class
            line = f"{class_id} {' '.join(f'{v:.6f}' for v in xywh)} {conf:.4f}"
            filtered_lines.append(line)

    # Optional: warn if too few detections
    if len(filtered_lines) < 3:
        print(f"[WARN] {fname}: only {len(filtered_lines)} boxes found")

    with open(label_path, "w") as f:
        f.write("\n".join(filtered_lines))

print("Smart auto-labeling finished.")
