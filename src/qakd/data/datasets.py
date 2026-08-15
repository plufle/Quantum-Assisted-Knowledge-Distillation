import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from qakd.data.splits import build_rps25_split, build_trashnet_split

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    if train:
        # RandAugment + random resized crop, fixed identically across every method (Trap #8)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class QAKDImageDataset(Dataset):
    """A split's (path, label) entries, resolved against a single root directory."""

    def __init__(self, root_dir, entries, transform):
        self.root_dir = root_dir
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(os.path.join(self.root_dir, entry["path"])).convert("RGB")
        return self.transform(image), entry["label"]


def _class_weighted_sampler(entries, num_classes):
    counts = [0] * num_classes
    for e in entries:
        counts[e["label"]] += 1
    class_weights = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_weights = [class_weights[e["label"]] for e in entries]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_trashnet_loaders(cfg):
    """cfg: the `dataset/trashnet` Hydra config node."""
    split = build_trashnet_split(
        cfg.data_dir, cfg.split.cache_path, tuple(cfg.split.ratios), cfg.split.seed
    )
    train_ds = QAKDImageDataset(cfg.data_dir, split["train"], build_transforms(cfg.image_size, train=True))
    val_ds = QAKDImageDataset(cfg.data_dir, split["val"], build_transforms(cfg.image_size, train=False))
    test_ds = QAKDImageDataset(cfg.data_dir, split["test"], build_transforms(cfg.image_size, train=False))

    # Trap #4: TrashNet class imbalance — weighted sampling instead of plain shuffle
    sampler = _class_weighted_sampler(split["train"], cfg.num_classes)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_rps25_loaders(cfg):
    """cfg: the `dataset/rps_25` Hydra config node."""
    split = build_rps25_split(cfg.data_dir, cfg.split.cache_path, cfg.train_fraction, cfg.split.seed)

    def _folder_entries(subdir):
        # RPS is inconsistent between splits: `test/` is nested by class, but
        # `validation/` is flat with class-prefixed filenames (e.g. "rock1.png").
        root = os.path.join(cfg.data_dir, subdir)
        class_names = split["class_names"]
        entries = []
        if os.path.isdir(os.path.join(root, class_names[0])):
            for label, cls in enumerate(class_names):
                for fname in sorted(os.listdir(os.path.join(root, cls))):
                    entries.append({"path": f"{subdir}/{cls}/{fname}", "label": label})
        else:
            for fname in sorted(os.listdir(root)):
                label = next(i for i, cls in enumerate(class_names) if fname.lower().startswith(cls))
                entries.append({"path": f"{subdir}/{fname}", "label": label})
        return entries

    train_ds = QAKDImageDataset(cfg.data_dir, split["train"], build_transforms(cfg.image_size, train=True))
    val_ds = QAKDImageDataset(
        cfg.data_dir, _folder_entries("validation"), build_transforms(cfg.image_size, train=False)
    )
    test_ds = QAKDImageDataset(
        cfg.data_dir, _folder_entries("test"), build_transforms(cfg.image_size, train=False)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
