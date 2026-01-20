""" 
Custom Python script using which the YOLO model was tested using the following dataset: 
https://drive.google.com/file/d/1Tw173zx3rGKYO46V7AZtiWgg4I5pjdxV/view?usp=drive_link
"""

import os
from collections import Counter
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "PATH"   # adjust
TEST_IMAGES = r"../test/images"
TEST_LABELS = r"../test/labels"

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]
# ----------------------------------------

def read_gt_class(label_path):
    """Read first class from YOLO label file (single-class image assumption)"""
    with open(label_path, "r") as f:
        lines = f.readlines()
    classes = [int(l.split()[0]) for l in lines]
    return Counter(classes).most_common(1)[0][0]

def predict_class(model, image_path):
    """Return predicted class index or None if no detection"""
    results = model(image_path, conf=0.25, iou=0.5, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    preds = results.boxes.cls.cpu().numpy().astype(int)
    return Counter(preds).most_common(1)[0][0]

def main():
    model = YOLO(MODEL_PATH)

    y_true = []
    y_pred = []

    images = [f for f in os.listdir(TEST_IMAGES) if f.lower().endswith(".jpg")]

    for img in tqdm(images, desc="Evaluating"):
        img_path = os.path.join(TEST_IMAGES, img)
        label_path = os.path.join(TEST_LABELS, img.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        gt_class = read_gt_class(label_path)
        pred_class = predict_class(model, img_path)

        if pred_class is None:
            pred_class = -1   # mark as "no detection"

        y_true.append(gt_class)
        y_pred.append(pred_class)

    # Remove "no detection" if desired
    valid_idx = [i for i,v in enumerate(y_pred) if v != -1]
    y_true_f = [y_true[i] for i in valid_idx]
    y_pred_f = [y_pred[i] for i in valid_idx]

    print("\nAccuracy:", accuracy_score(y_true_f, y_pred_f))

    print("\nClassification Report:")
    print(classification_report(y_true_f, y_pred_f, target_names=CLASS_NAMES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_f, y_pred_f))

if __name__ == "__main__":
    main()

