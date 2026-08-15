import json
import os

from sklearn.model_selection import train_test_split

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _list_images(class_dir):
    return sorted(f for f in os.listdir(class_dir) if f.lower().endswith(_IMAGE_EXTS))


def build_trashnet_split(data_dir, cache_path, ratios=(0.70, 0.15, 0.15), seed=0):
    """Stratified 70/15/15 split for TrashNet (no official split exists).
    Cached once to `cache_path` and never regenerated — see CLAUDE.md."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        for fname in _list_images(os.path.join(data_dir, cls)):
            paths.append(f"{cls}/{fname}")
            labels.append(label)

    train_frac, val_frac, test_frac = ratios
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, train_size=train_frac, stratify=labels, random_state=seed
    )
    val_share = val_frac / (val_frac + test_frac)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, train_size=val_share, stratify=rest_labels, random_state=seed
    )

    split = {
        "seed": seed,
        "ratios": {"train": train_frac, "val": val_frac, "test": test_frac},
        "class_names": class_names,
        "train": [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)],
        "val": [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)],
        "test": [{"path": p, "label": l} for p, l in zip(test_paths, test_labels)],
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def build_rps25_split(data_dir, cache_path, fraction=0.25, seed=0):
    """Stratified 25%-of-train subsample for RPS. `validation`/`test` folders are used
    in full, unsubsampled — the low-data regime only applies to training (CLAUDE.md)."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    train_dir = os.path.join(data_dir, "train")
    class_names = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        for fname in _list_images(os.path.join(train_dir, cls)):
            paths.append(f"train/{cls}/{fname}")
            labels.append(label)

    kept_paths, _, kept_labels, _ = train_test_split(
        paths, labels, train_size=fraction, stratify=labels, random_state=seed
    )

    split = {
        "seed": seed,
        "train_fraction": fraction,
        "class_names": class_names,
        "train": [{"path": p, "label": l} for p, l in zip(kept_paths, kept_labels)],
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


if __name__ == "__main__":
    trashnet = build_trashnet_split("data/raw/TrashNet", "data/splits/trashnet.json")
    rps = build_rps25_split("data/raw/Rock-Paper-Scissors", "data/splits/rps_25.json")
    print(
        f"trashnet: {len(trashnet['train'])} train / "
        f"{len(trashnet['val'])} val / {len(trashnet['test'])} test"
    )
    print(f"rps_25:   {len(rps['train'])} train (25% subsample)")
