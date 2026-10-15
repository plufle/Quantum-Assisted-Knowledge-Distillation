"""Real-world Rock-Paper-Scissors data: webcam captures (data/raw/rps_real) merged with the full
rendered RPS training set (data/raw/Rock-Paper-Scissors). Separate from the QAKD research setup.

Classes: paper, rock, scissors (RPS order) + none. Images are classified whole-frame: a webcam
frame is resized to 128x128 without cropping (hands often sit near the frame edge).

Split (session s1 only, by the user's choice): each class's capture sequence is cut into blocks of
10 consecutive frames (one block ~ one spot); frames 20-29 -> validation, 40-49 -> test, the rest
-> train, minus the frame either side of a held-out block (photos 1 s apart are near-duplicates).
Same room and sitting for all three, so real-world scores from this split are optimistic.

Training-time augmentation, aimed at the rendered-vs-real gap: with probability COMPOSITE_P a
rendered RPS hand is cut out of its white background and pasted, at a random scale and position,
onto a 4:3 crop of a real 'none' TRAIN frame (never a validation/test frame); then crops,
flips, rotation, colour/lighting jitter and blur are applied to every image.
"""
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_ROOT = os.path.join(REPO, "data", "raw", "rps_real")
RPS_ROOT = os.path.join(REPO, "data", "raw", "Rock-Paper-Scissors")
SPLIT_PATH = os.path.join(REPO, "realworld", "splits", "rps_real_s1.json")
CLASSES = ["paper", "rock", "scissors", "none"]
IMAGE_SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# frames with no visible gesture (taken as SPACE was pressed, or fully blurred) — reviewed by eye
EXCLUDE = {"s1": {"rock": [0], "paper": [0], "scissors": [0, 34, 35, 41], "none": []}}
VAL_BLOCK, TEST_BLOCK, BLOCK = 2, 4, 10
REAL_SHARE = 0.35      # fraction of each training epoch drawn from real photos
COMPOSITE_P = 0.7      # rendered hand pasted onto a real background


def build_split(session="s1"):
    """Write the frame-level split JSON (idempotent) and return it."""
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            return json.load(f)
    split = {"train": [], "val": [], "test": [], "excluded": [],
             "rule": f"blocks of {BLOCK} frames; block {VAL_BLOCK} val, block {TEST_BLOCK} test, "
                     "frames adjacent to a held-out block dropped from train"}
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(REAL_ROOT, session, cls, "*.jpg")))
        for f in files:
            idx = int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])
            rel = os.path.relpath(f, REAL_ROOT).replace("\\", "/")
            entry = {"path": rel, "label": CLASSES.index(cls)}
            block = idx // BLOCK
            if idx in EXCLUDE[session][cls]:
                split["excluded"].append(rel)
            elif block == VAL_BLOCK:
                split["val"].append(entry)
            elif block == TEST_BLOCK:
                split["test"].append(entry)
            elif idx in {VAL_BLOCK * BLOCK - 1, (VAL_BLOCK + 1) * BLOCK, TEST_BLOCK * BLOCK - 1, (TEST_BLOCK + 1) * BLOCK}:
                split["excluded"].append(rel)  # guard frame next to a held-out block
            else:
                split["train"].append(entry)
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=1)
    return split


def rps_entries(subdir):
    return [{"path": p, "label": CLASSES.index(cls)}
            for cls in CLASSES[:3] for p in sorted(glob.glob(os.path.join(RPS_ROOT, subdir, cls, "*.png")))]


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(1.0, 1.6)),  # 4:3 frames included whole
        T.RandomHorizontalFlip(),
        T.RandomRotation(20, fill=0),
        T.ColorJitter(0.4, 0.4, 0.4, 0.05),
        T.RandomGrayscale(0.05),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    """Whole frame, squashed to 128x128 — exactly what the live demo does for this model."""
    return T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def _rgb(path):
    return Image.open(path).convert("RGB")


class RealFrames(Dataset):
    def __init__(self, entries, transform):
        self.entries, self.transform = entries, transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        return self.transform(_rgb(os.path.join(REAL_ROOT, e["path"]))), e["label"]


class RenderedRPS(Dataset):
    """Rendered RPS hands; during training, most are composited onto real background frames."""

    def __init__(self, entries, transform, backgrounds=None, composite_p=0.0):
        self.entries, self.transform = entries, transform
        self.backgrounds, self.composite_p = backgrounds or [], composite_p
        self._bg_cache = {}

    def __len__(self):
        return len(self.entries)

    def _background(self):
        path = random.choice(self.backgrounds)
        if path not in self._bg_cache:
            self._bg_cache[path] = np.asarray(_rgb(path))
        bg = self._bg_cache[path]
        h, w = bg.shape[:2]
        s = random.uniform(0.6, 1.0)
        ch, cw = int(h * s), int(w * s)
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        return cv2.resize(bg[y:y + ch, x:x + cw], (400, 300), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _hand_mask(img):
        """Foreground = not (bright and unsaturated) background connected to the image border."""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bg = ((hsv[..., 2] > 200) & (hsv[..., 1] < 40)).astype(np.uint8)
        n, lab = cv2.connectedComponents(bg)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        bg_conn = np.isin(lab, list(border)).astype(np.uint8)
        fg = 1 - bg_conn
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return cv2.GaussianBlur(fg.astype(np.float32), (5, 5), 0)  # feathered alpha

    def _composite(self, img):
        canvas = self._background().astype(np.float32)
        side = int(random.uniform(0.55, 1.0) * 300)
        hand = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        alpha = cv2.resize(self._hand_mask(img), (side, side))[..., None]
        y, x = random.randint(0, 300 - side), random.randint(0, 400 - side)
        roi = canvas[y:y + side, x:x + side]
        canvas[y:y + side, x:x + side] = alpha * hand + (1 - alpha) * roi
        return Image.fromarray(canvas.clip(0, 255).astype(np.uint8))

    def __getitem__(self, i):
        e = self.entries[i]
        img = _rgb(e["path"])
        if self.backgrounds and random.random() < self.composite_p:
            img = self._composite(np.asarray(img))
        return self.transform(img), e["label"]


def build_loaders(batch_size, num_workers=4, seed=0):
    split = build_split()
    backgrounds = [os.path.join(REAL_ROOT, e["path"]) for e in split["train"] if e["label"] == CLASSES.index("none")]
    rendered = RenderedRPS(rps_entries("train"), train_transform(), backgrounds, COMPOSITE_P)
    real = RealFrames(split["train"], train_transform())
    train_ds = ConcatDataset([rendered, real])
    weights = [(1 - REAL_SHARE) / len(rendered)] * len(rendered) + [REAL_SHARE / len(real)] * len(real)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    common = {"num_workers": num_workers, "persistent_workers": num_workers > 0}
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **common),
        "val": DataLoader(RealFrames(split["val"], eval_transform()), batch_size=64, **common),
        "test": DataLoader(RealFrames(split["test"], eval_transform()), batch_size=64, **common),
        "rps_test": DataLoader(RenderedRPS(rps_entries("test"), eval_transform()), batch_size=64, **common),
    }
    sizes = {"train_rendered": len(rendered), "train_real": len(real), "val": len(split["val"]),
             "test": len(split["test"]), "rps_test": len(loaders["rps_test"].dataset),
             "backgrounds": len(backgrounds), "excluded": len(split["excluded"])}
    return loaders, sizes
