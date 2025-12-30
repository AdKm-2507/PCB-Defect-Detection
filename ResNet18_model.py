from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import shutil

DATA_ROOT = Path("train")         
OUT_ROOT  = Path("dataset_split")    
VAL_RATIO = 0.2
IMG_SIZE = 224
EPOCHS_HEAD = 5
EPOCHS_FINE = 15

# --------------------------------------------------------------
# 1) collect image paths + labels + design groups
# --------------------------------------------------------------
image_paths, labels, groups = [], [], []

for cls_dir in sorted(DATA_ROOT.iterdir()):
    if not cls_dir.is_dir():
        continue

    cls_name = cls_dir.name

    for img_path in cls_dir.glob("*.jpg"):
        fname = img_path.name
        design_id = fname.split("_")[0]     # PCB design ID

        image_paths.append(str(img_path))
        labels.append(cls_name)
        groups.append(design_id)

print(f"Total images collected: {len(image_paths)}")

# --------------------------------------------------------------
# 2) group-safe split
# --------------------------------------------------------------
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VAL_RATIO,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(image_paths, labels, groups=groups)
)

train_files = [image_paths[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]

val_files   = [image_paths[i] for i in val_idx]
val_labels  = [labels[i] for i in val_idx]

print(f"Train images: {len(train_files)}")
print(f"Val images:   {len(val_files)}")


# --------------------------------------------------------------
# 3) write split to dataset_split/ (train/val/cls)
# --------------------------------------------------------------
def copy_split(files, labels, split_name):
    split_root = OUT_ROOT / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    for src, cls in zip(files, labels):
        dst_dir = split_root / cls
        dst_dir.mkdir(exist_ok=True)

        dst = dst_dir / Path(src).name
        shutil.copy2(src, dst)

    print(f"wrote {len(files)} files to {split_name}/")

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

copy_split(train_files, train_labels, "train")
copy_split(val_files,   val_labels,   "val")

print("Done — split written to:", OUT_ROOT.resolve())

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch

# -------------------------------------------------------------
# 1) build class index mapping from folder names
# -------------------------------------------------------------
classes = sorted([d.name for d in (OUT_ROOT / "train").iterdir() if d.is_dir()])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

print("Classes:", classes)


# -------------------------------------------------------------
# 2) custom dataset
# -------------------------------------------------------------
import random
import numpy as np

class PCBPatchDataset(Dataset):
    """
    Dataset that loads PCB images and samples edge-rich patches.
    Used because defect locations are unknown (weak labels).

    """
    def __init__(self, root, split, patches_per_image=6, train=True):
        self.root = Path(root) / split
        self.train = train
        self.patches_per_image = patches_per_image
        self.samples = []

        for cls in classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), class_to_idx[cls]))

        print(f"{split} samples:", len(self.samples))

        self.base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            )
        ])

    def _sample_patch(self, img):
        """
        Randomly sample several crops and prefer the most
        structure-rich (Sobel-edge) patch.

        """
        h, w = img.shape[:2]

        for _ in range(18):
            scale = random.uniform(0.50, 0.80)
            size = int(min(h, w) * scale)

            top  = random.randint(0, h - size)
            left = random.randint(0, w - size)

            patch = img[top:top+size, left:left+size]

            # prefer patches with structure
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_strength = np.mean(np.abs(gx) + np.abs(gy))

            if edge_strength > 12:   #tuneable but works well
                return cv2.resize(patch, (IMG_SIZE, IMG_SIZE))

        # fallback
        return cv2.resize(patch, (IMG_SIZE, IMG_SIZE))



    def _augment(self, img):
        #Light augmentation: rotation, flip, brightness.

        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        if random.random() < 0.4:
            alpha = random.uniform(0.9, 1.1)
            beta  = random.randint(-12, 12)
            img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # create several patches and pick one at random
        candidates = [self._sample_patch(img) for _ in range(self.patches_per_image)]

        # score patches by edge richness
        def score(p):
            gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            return float(np.mean(np.abs(gx) + np.abs(gy)))

        patch = max(candidates, key=score)


        if self.train:
            patch = self._augment(patch)

        patch = self.base_tf(patch)
        return patch, label

    def __len__(self):
        return len(self.samples)

# -------------------------------------------------------------
# 3) build datasets & dataloaders
# -------------------------------------------------------------
train_ds = PCBPatchDataset(OUT_ROOT, "train", train=True)
val_ds   = PCBPatchDataset(OUT_ROOT, "val", train=False)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,      
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,     
    num_workers=0,
    pin_memory=True
)

import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

NUM_CLASSES = len(classes)

# load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# replace classifier head
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()


# Phase 1: Freeze backbone, only train classifier
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

def train_one_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0
    correct = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()

    val_loss = total_loss / len(val_loader.dataset)
    val_acc  = correct / len(val_loader.dataset)

    return val_loss, val_acc

if __name__ == "__main__":
    print("\n=== Phase 1: train classifier head ===")
    for epoch in range(1, EPOCHS_HEAD + 1):
        tr = train_one_epoch()
        va, acc = validate()

        print(f"[Head {epoch:02d}] train={tr:.4f}  val={va:.4f}  acc={acc:.4f}")

    print("\n=== Phase 2: fine-tune backbone ===")

    # Phase 2: fine-tune deeper layers at low LR
    for name, p in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            p.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=5e-4
    )

    for epoch in range(1, EPOCHS_FINE + 1):
        tr = train_one_epoch()
        va, acc = validate()

        print(f"[Fine {epoch:02d}] train={tr:.4f}  val={va:.4f}  acc={acc:.4f}")
        from collections import Counter

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.tolist())

        print("Pred dist:", Counter(all_preds))
