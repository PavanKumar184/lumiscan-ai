# %% [markdown]
# # NIH ChestX-ray14 DR-Ultra Training Notebook
#
# Project: **A Vision-Language Assisted DR-Ultra Framework for Robust Thoracic Disease Detection from Imbalanced Chest Radiographs**
#
# This Kaggle-ready notebook script trains an imbalance-aware multi-label chest X-ray classifier on the NIH ChestX-ray14 dataset.
#
# Dataset link:
# https://www.kaggle.com/datasets/nih-chest-xrays/data
#
# What this notebook does:
# - Detects the Kaggle dataset path automatically.
# - Loads `Data_Entry_2017.csv`.
# - Maps image names to actual image paths.
# - Cleans invalid image entries.
# - Converts disease labels into 14-class multi-hot vectors.
# - Splits data into train/validation/test.
# - Computes class weights for imbalanced learning.
# - Trains the DR-Ultra image branch with a selectable visual backbone.
# - DenseNet121, ResNet50, and EfficientNet-B0 are used as visual encoder backbones
#   inside the DR-Ultra framework, not as the main proposed model.
# - Saves the best model checkpoint.
# - Creates a downloadable `.zip` file containing the model and class metadata.
#
# Notes:
# - Use Kaggle GPU runtime for practical training.
# - For a stronger final model, increase `EPOCHS` to 10-20.
# - This script uses only PyTorch, Torchvision, Pandas, NumPy, and Scikit-learn.

# %%
import os
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


# %% [markdown]
# ## Configuration

# %%
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
VISUAL_BACKBONE = "densenet121"  # options: densenet121, resnet50, efficientnet_b0
THRESHOLD = 0.5
USE_PRETRAINED = False  # Keep False on Kaggle if internet is disabled.

OUTPUT_DIR = Path("/kaggle/working/dr_ultra_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# %% [markdown]
# ## Reproducibility

# %%
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# %% [markdown]
# ## Class Labels
#
# NIH ChestX-ray14 has 14 disease classes. `No Finding` is kept for statistics but is not used as a positive disease class.

# %%
NIH_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(NIH_CLASSES)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(NIH_CLASSES)}


# %% [markdown]
# ## Dataset Path Auto-Detection
#
# Kaggle dataset folder names may differ depending on how the dataset is attached. This function searches `/kaggle/input` and finds the folder containing `Data_Entry_2017.csv`.

# %%
def find_nih_dataset_root(input_root: str = "/kaggle/input") -> Path:
    input_path = Path(input_root)
    candidates = []

    for csv_path in input_path.rglob("Data_Entry_2017.csv"):
        if csv_path.is_file():
            candidates.append(csv_path.parent)

    if not candidates:
        raise FileNotFoundError(
            "Could not find Data_Entry_2017.csv. "
            "Please attach the Kaggle dataset: nih-chest-xrays/data"
        )

    candidates = sorted(candidates, key=lambda p: len(str(p)))
    print("Detected dataset root:", candidates[0])
    return candidates[0]


DATASET_ROOT = find_nih_dataset_root()
CSV_PATH = DATASET_ROOT / "Data_Entry_2017.csv"


# %% [markdown]
# ## Build Image Index
#
# The full NIH dataset contains image folders such as `images_001/images`, `images_002/images`, etc.

# %%
def build_image_index(dataset_root: Path) -> Dict[str, str]:
    image_index = {}
    image_paths = list(dataset_root.rglob("*.png"))
    print(f"Found PNG files: {len(image_paths)}")

    for path in image_paths:
        image_index[path.name] = str(path)

    return image_index


image_index = build_image_index(DATASET_ROOT)


# %% [markdown]
# ## Load Metadata, Clean Paths, and Multi-Hot Encode Labels

# %%
def normalize_labels(label_string: str) -> List[str]:
    labels = [label.strip() for label in str(label_string).split("|") if label.strip()]
    labels = [label for label in labels if label != "No Finding"]
    return labels


def encode_multi_hot(labels: Sequence[str]) -> np.ndarray:
    encoded = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label in labels:
        if label in CLASS_TO_INDEX:
            encoded[CLASS_TO_INDEX[label]] = 1.0
    return encoded


metadata = pd.read_csv(CSV_PATH)
print("Original CSV rows:", len(metadata))

metadata["image_path"] = metadata["Image Index"].map(image_index)
metadata = metadata.dropna(subset=["image_path"]).copy()
print("Rows with available images:", len(metadata))

metadata["label_list"] = metadata["Finding Labels"].apply(normalize_labels)
metadata["num_labels"] = metadata["label_list"].apply(len)
metadata["has_finding"] = (metadata["num_labels"] > 0).astype(int)
metadata["label_signature"] = metadata["label_list"].apply(
    lambda labels: "|".join(sorted(labels)) if labels else "No Finding"
)

encoded = np.vstack(metadata["label_list"].apply(encode_multi_hot).to_numpy())
encoded_df = pd.DataFrame(encoded, columns=NIH_CLASSES, index=metadata.index)
metadata = pd.concat([metadata, encoded_df], axis=1).reset_index(drop=True)

print(metadata[["Image Index", "Finding Labels", "image_path", "num_labels"]].head())


# %% [markdown]
# ## Dataset Statistics

# %%
class_counts_all = metadata[NIH_CLASSES].sum().astype(int)
class_freq_all = class_counts_all / len(metadata)

print("Total usable samples:", len(metadata))
print("No Finding samples:", int((metadata["num_labels"] == 0).sum()))
print("Disease-positive samples:", int((metadata["num_labels"] > 0).sum()))
print("\nClass distribution:")
print(pd.DataFrame({"positive_count": class_counts_all, "frequency": class_freq_all}).to_string())


# %% [markdown]
# ## Multi-Label-Aware Train/Validation/Test Split
#
# This is a lightweight greedy split that tries to preserve class distribution in multi-label data without requiring extra packages.

# %%
def multilabel_split(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    labels = df[list(label_cols)].to_numpy(dtype=np.int64)
    n = len(df)

    train_target = int(round(n * train_ratio))
    val_target = int(round(n * val_ratio))
    test_target = n - train_target - val_target

    split_names = ["train", "val", "test"]
    target_sizes = {"train": train_target, "val": val_target, "test": test_target}
    current_sizes = {name: 0 for name in split_names}

    target_label_counts = {
        "train": labels.sum(axis=0) * train_ratio,
        "val": labels.sum(axis=0) * val_ratio,
        "test": labels.sum(axis=0) * (1.0 - train_ratio - val_ratio),
    }
    current_label_counts = {name: np.zeros(len(label_cols), dtype=np.float64) for name in split_names}
    assigned = {name: [] for name in split_names}

    cardinality = labels.sum(axis=1)
    noise = rng.random(n)
    order = np.lexsort((noise, -cardinality))

    for idx in order:
        y = labels[idx]
        pos = np.where(y == 1)[0]
        available = [name for name in split_names if current_sizes[name] < target_sizes[name]]

        best_name = None
        best_score = -1e18
        for name in available:
            size_deficit = target_sizes[name] - current_sizes[name]
            label_deficit = np.clip(target_label_counts[name] - current_label_counts[name], 0, None)
            score = float(label_deficit[pos].sum()) if len(pos) else 0.0
            score += 0.05 * size_deficit
            if score > best_score:
                best_score = score
                best_name = name

        assigned[best_name].append(idx)
        current_sizes[best_name] += 1
        current_label_counts[best_name] += y

    train_df = df.iloc[assigned["train"]].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = df.iloc[assigned["val"]].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = df.iloc[assigned["test"]].sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


train_df, val_df, test_df = multilabel_split(metadata, NIH_CLASSES, seed=SEED)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


# %% [markdown]
# ## Compute Class Weights for DR-Ultra-Style Imbalance Handling
#
# Main formula:
#
# \[
# pos\_weight_i = \frac{N_i^-}{N_i^+}
# \]
#
# Rare classes get higher weights.

# %%
def compute_class_weights(df: pd.DataFrame, label_cols: Sequence[str]) -> Tuple[pd.DataFrame, torch.Tensor]:
    y = df[list(label_cols)].to_numpy(dtype=np.float32)
    n = float(len(df))
    positive = y.sum(axis=0)
    negative = n - positive
    frequency = positive / max(n, 1.0)

    pos_weight = negative / np.clip(positive, 1.0, None)
    inv_frequency = 1.0 / np.clip(frequency, 1e-8, None)
    inv_frequency = inv_frequency / inv_frequency.mean()

    weights_df = pd.DataFrame({
        "class_name": label_cols,
        "positive_count": positive.astype(int),
        "negative_count": negative.astype(int),
        "frequency": frequency,
        "inverse_frequency_weight": inv_frequency,
        "bce_pos_weight": pos_weight,
    })
    return weights_df, torch.tensor(pos_weight, dtype=torch.float32)


weights_df, POS_WEIGHT = compute_class_weights(train_df, NIH_CLASSES)
weights_df.to_csv(OUTPUT_DIR / "class_weights.csv", index=False)
print(weights_df.to_string(index=False))


# %% [markdown]
# ## Export Cleaned Metadata

# %%
metadata.to_csv(OUTPUT_DIR / "cleaned_metadata.csv", index=False)
train_df.to_csv(OUTPUT_DIR / "train_metadata.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val_metadata.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_metadata.csv", index=False)

with open(OUTPUT_DIR / "classes.json", "w") as f:
    json.dump(NIH_CLASSES, f, indent=2)

print("Metadata and class files saved to:", OUTPUT_DIR)


# %% [markdown]
# ## PyTorch Dataset and Transforms

# %%
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class NIHChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols: Sequence[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = list(label_cols)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            # Rare fallback in case an image becomes unreadable.
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        return image, labels


train_dataset = NIHChestXrayDataset(train_df, NIH_CLASSES, transform=train_transform)
val_dataset = NIHChestXrayDataset(val_df, NIH_CLASSES, transform=eval_transform)
test_dataset = NIHChestXrayDataset(test_df, NIH_CLASSES, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)


# %% [markdown]
# ## Build DR-Ultra Model
#
# DR-Ultra is the main framework. In this training notebook, the implemented
# trainable component is the image branch of DR-Ultra with imbalance-aware loss.
#
# DenseNet121, ResNet50, and EfficientNet-B0 are used only as interchangeable
# visual encoder backbones. Later, the language encoder and fusion module can be
# attached on top of the same training structure.

# %%
class DRUltraImageBranch(nn.Module):
    """
    DR-Ultra image-branch model.

    The selected CNN backbone acts as the visual encoder. The imbalance-aware
    DR-Ultra behavior is introduced through the weighted multi-label objective.
    This class is intentionally structured so that a text encoder and fusion
    block can be added later without changing the dataset or training pipeline.
    """

    def __init__(self, visual_backbone: str, num_classes: int, use_pretrained: bool = False):
        super().__init__()
        self.visual_backbone_name = visual_backbone.lower()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.model = self._build_visual_branch(
            self.visual_backbone_name,
            num_classes,
            use_pretrained=use_pretrained,
        )

    def _build_visual_branch(
        self,
        visual_backbone: str,
        num_classes: int,
        use_pretrained: bool,
    ) -> nn.Module:
        if visual_backbone == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.densenet121(weights=weights)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(model.classifier.in_features, num_classes),
            )
            return model

        if visual_backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(model.fc.in_features, num_classes),
            )
            return model

        if visual_backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model

        raise ValueError(f"Unsupported visual backbone: {visual_backbone}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def build_dr_ultra_model(
    visual_backbone: str,
    num_classes: int,
    use_pretrained: bool = False,
) -> nn.Module:
    return DRUltraImageBranch(
        visual_backbone=visual_backbone,
        num_classes=num_classes,
        use_pretrained=use_pretrained,
    )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    # Backward-compatible helper for loading checkpoints later.
    return build_dr_ultra_model(
        visual_backbone=model_name,
        num_classes=num_classes,
        use_pretrained=False,
    )


"""
Legacy direct-backbone builder retained only for reference:
DenseNet121, ResNet50, and EfficientNet-B0 are not the proposed model by
themselves; they are visual encoders inside the DR-Ultra framework.
"""

def _build_backbone_reference(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier.in_features, num_classes),
        )
        return model

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


model = build_dr_ultra_model(
    VISUAL_BACKBONE,
    NUM_CLASSES,
    use_pretrained=USE_PRETRAINED,
).to(DEVICE)
print("Main framework: DR-Ultra")
print("Visual backbone:", VISUAL_BACKBONE)
print("Use ImageNet pretrained weights:", USE_PRETRAINED)
print(model.__class__.__name__)


# %% [markdown]
# ## Loss, Optimizer, and Scheduler

# %%
criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    verbose=True,
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


# %% [markdown]
# ## Metrics

# %%
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "macro_auc": safe_auc(y_true, y_prob),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


# %% [markdown]
# ## Train and Validation Functions

# %%
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        if step % 200 == 0:
            print(f"Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)
        running_loss += loss.item() * images.size(0)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


# %% [markdown]
# ## Training Loop
#
# Best checkpoint is selected using validation macro AUC.

# %%
best_val_auc = -1.0
history = []
best_model_path = OUTPUT_DIR / f"best_dr_ultra_{VISUAL_BACKBONE}_nih.pth"

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
    val_metrics = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_metrics["macro_auc"])

    epoch_log = {
        "epoch": epoch,
        "train_loss": train_loss,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "lr": optimizer.param_groups[0]["lr"],
        "time_sec": time.time() - start,
    }
    history.append(epoch_log)

    print("\nEpoch", epoch)
    print(json.dumps(epoch_log, indent=2))

    if val_metrics["macro_auc"] > best_val_auc:
        best_val_auc = val_metrics["macro_auc"]
        torch.save({
            "framework": "DR-Ultra",
            "visual_backbone": VISUAL_BACKBONE,
            "model_name": VISUAL_BACKBONE,
            "use_pretrained": USE_PRETRAINED,
            "model_state_dict": model.state_dict(),
            "classes": NIH_CLASSES,
            "image_size": IMAGE_SIZE,
            "threshold": THRESHOLD,
            "pos_weight": POS_WEIGHT.cpu().tolist(),
            "val_metrics": val_metrics,
            "epoch": epoch,
        }, best_model_path)
        print("Saved best model:", best_model_path)

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
print("Best validation macro AUC:", best_val_auc)


# %% [markdown]
# ## Test Evaluation

# %%
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
test_metrics = evaluate(model, test_loader, criterion, DEVICE)

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

print("Test metrics:")
print(json.dumps(test_metrics, indent=2))


# %% [markdown]
# ## Save Inference Metadata and Package for Download

# %%
inference_config = {
    "framework": "DR-Ultra",
    "visual_backbone": VISUAL_BACKBONE,
    "use_pretrained": USE_PRETRAINED,
    "num_classes": NUM_CLASSES,
    "classes": NIH_CLASSES,
    "image_size": IMAGE_SIZE,
    "threshold": THRESHOLD,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "note": "Use sigmoid on model logits for multi-label disease probabilities.",
}

with open(OUTPUT_DIR / "inference_config.json", "w") as f:
    json.dump(inference_config, f, indent=2)

zip_path = shutil.make_archive(
    base_name="/kaggle/working/dr_ultra_trained_model_package",
    format="zip",
    root_dir=str(OUTPUT_DIR),
)

print("Training complete.")
print("Best model:", best_model_path)
print("Download package:", zip_path)


# %% [markdown]
# ## How to Use the Saved Model Later
#
# ```python
# checkpoint = torch.load("best_dr_ultra_densenet121_nih.pth", map_location="cpu")
# model = build_dr_ultra_model(
#     checkpoint["visual_backbone"],
#     len(checkpoint["classes"]),
#     use_pretrained=False,
# )
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()
# ```
#
# During inference:
# ```python
# logits = model(image_tensor)
# probabilities = torch.sigmoid(logits)
# ```
